//! A 2D profile swept along an axis, as a first-class shape.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use crate::mass_properties::MassProperties;
use crate::math::{Matrix, Pose, Real, Vector, Vector2};
use crate::query::details::{
    local_point_projection_on_support_map, local_ray_intersection_with_support_map_with_params,
};
use crate::query::gjk::VoronoiSimplex;
use crate::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use crate::shape::{
    FeatureId, PackedFeatureId, PolygonalFeature, PolygonalFeatureMap, Shape, ShapeType, SupportMap,
    TypedShape,
};

/// A convex 2D profile in the XZ plane, extruded along Y.
///
/// **Why this exists rather than a `ConvexPolyhedron` of the same corners.** A prism built as a
/// convex hull is a *mesh*: its surface normals are a fixed, finite set, so as a body rotates the
/// contact normal jumps from facet to facet. That is visible as roughness — a 16-sided bar binds in
/// a 16-stave bore where a `Cylinder` of the same size runs freely. A shape defined by its **support
/// function** has no such quantisation along the sweep: the extrusion direction is exact, and only
/// the profile is discrete, which is honest because the profile really is a polygon.
///
/// It also sidesteps the four-vertex truncation in [`crate::shape::ConvexPolyhedron`]'s support
/// feature, because the cap face is generated here rather than looked up from a face list.
///
/// The profile must be **convex** and wound counter-clockwise. Concave parts are the caller's job to
/// decompose — the same rule `Compound` already imposes.
///
/// The Y axis is the sweep, matching [`crate::shape::Cylinder`], so the two are interchangeable in
/// a scene without re-orienting anything.
/// A circular arc on a profile's boundary, swept counter-clockwise from `start` to `end`.
///
/// This is what makes a *drawn* curve stay curved. A profile of straight segments has a finite set
/// of surface normals, so a body rolling on it steps from facet to facet; an arc's normal is
/// `center + radius * dir`, exact for every direction it spans. The arc's own endpoints live in the
/// profile's vertex list, so a direction outside the span still finds them.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProfileArc {
    /// Centre of the circle the arc lies on.
    pub center: Vector2,
    pub radius: Real,
    /// Start angle, radians.
    pub start: Real,
    /// End angle, radians, counter-clockwise from `start`. `end - start` is the swept angle and
    /// must be in `(0, 2π]`; a full turn is a circle with no straight edges at all.
    pub end: Real,
}

impl ProfileArc {
    /// A full circle.
    pub fn circle(center: Vector2, radius: Real) -> Self {
        Self { center, radius, start: 0.0, end: core::f32::consts::TAU as Real }
    }

    /// Swept angle, radians.
    pub fn sweep(&self) -> Real {
        self.end - self.start
    }

    /// Does this arc face `angle`? Convexity means it bulges outward, so the outward normal at a
    /// point is the radial direction — and the arc supports exactly the directions it spans.
    fn spans(&self, angle: Real) -> bool {
        let tau = core::f32::consts::TAU as Real;
        let mut a = angle - self.start;
        while a < 0.0 {
            a += tau;
        }
        while a >= tau {
            a -= tau;
        }
        a <= self.sweep()
    }

    fn at(&self, angle: Real) -> Vector2 {
        self.center + Vector2::new(angle.cos(), angle.sin()) * self.radius
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtrudedProfile {
    /// Convex boundary loop in the XZ plane, counter-clockwise. May be empty if the profile is a
    /// single closed arc.
    pub profile: Vec<Vector2>,
    /// Circular arcs on the boundary, which stay exactly circular rather than being tessellated.
    pub arcs: Vec<ProfileArc>,
    /// Half the extrusion length, along Y.
    pub half_depth: Real,
}

impl ExtrudedProfile {
    /// A straight-edged profile (XZ, convex, counter-clockwise) swept `half_depth * 2` along Y.
    pub fn new(profile: Vec<Vector2>, half_depth: Real) -> Self {
        Self { profile, arcs: Vec::new(), half_depth }
    }

    /// A profile mixing straight edges and circular arcs — a rounded rectangle, a filleted bracket,
    /// a slot with radiused ends. Sharp where it was drawn sharp, exactly round where it was drawn
    /// round, which no single primitive can express.
    pub fn with_arcs(profile: Vec<Vector2>, arcs: Vec<ProfileArc>, half_depth: Real) -> Self {
        Self { profile, arcs, half_depth }
    }

    /// A true circular bar: one full arc, no vertices, no facets anywhere.
    pub fn round(radius: Real, half_depth: Real) -> Self {
        Self::with_arcs(
            Vec::new(),
            alloc::vec![ProfileArc::circle(Vector2::ZERO, radius)],
            half_depth,
        )
    }

    /// The boundary as a polygon, arcs sampled. Used where a *shape* is needed rather than a
    /// support direction: the contact cap feature, and the inertia integral.
    ///
    /// Never for the support function or the centroid, both of which stay exact.
    fn tessellated(&self) -> Vec<Vector2> {
        /// Samples per full turn. Only ever affects the inertia (by ~1e-5 relative) and which four
        /// points a cap feature picks, neither of which is sensitive.
        const PER_TURN: usize = 256;
        if self.arcs.is_empty() {
            return self.profile.clone();
        }
        let tau = core::f32::consts::TAU as Real;
        let mut out = self.profile.clone();
        for arc in &self.arcs {
            let n = ((arc.sweep() / tau) * PER_TURN as Real).ceil().max(2.0) as usize;
            for k in 0..n {
                let t = k as Real / n as Real;
                out.push(arc.at(arc.start + arc.sweep() * t));
            }
        }
        // Sort by angle about the centroid of the samples, which is enough to wind a convex loop.
        let c = out.iter().fold(Vector2::ZERO, |a, b| a + *b) / out.len() as Real;
        out.sort_by(|a, b| {
            let (pa, pb) = ((a.y - c.y).atan2(a.x - c.x), (b.y - c.y).atan2(b.x - c.x));
            pa.partial_cmp(&pb).unwrap_or(core::cmp::Ordering::Equal)
        });
        out
    }

    /// A regular `sides`-gon of the given radius across the corners.
    pub fn regular(radius: Real, sides: usize, half_depth: Real) -> Self {
        let n = sides.max(3);
        let profile = (0..n)
            .map(|i| {
                let a = i as Real / n as Real * Real::from(2.0) * core::f32::consts::PI as Real;
                Vector2::new(radius * a.cos(), radius * a.sin())
            })
            .collect();
        Self::new(profile, half_depth)
    }

}

/// Round a centroid that is *indistinguishably* off-centre to exactly on-centre.
///
/// Summing a symmetric profile's moments leaves a residue — 1e-23 on a 0.25 m circle — which is not
/// an offset, it is the arithmetic failing to cancel. Reporting it as an offset would be wrong on
/// its own terms, and it happens to be catastrophic in practice: Avian 0.7's rigid joints diverge
/// when a body's centre of mass is non-zero by *any* amount at all.
///
/// The threshold is relative to the profile's own size, so it means "below the resolution this sum
/// can resolve" rather than a fixed number of metres. A genuinely off-centre profile — an
/// L-bracket, say — is orders of magnitude above it and passes through untouched.
fn snap_to_centre(c: Vector2, area: Real) -> Vector2 {
    let scale = area.abs().sqrt().max(Real::MIN_POSITIVE);
    let eps = scale * 1.0e-7;
    Vector2::new(
        if c.x.abs() < eps { 0.0 } else { c.x },
        if c.y.abs() < eps { 0.0 } else { c.y },
    )
}

/// Area and centroid of a closed polygon, exactly — Green's theorem over its boundary.
fn polygon_area_centroid(poly: &[Vector2]) -> (Real, Vector2) {
    let n = poly.len();
    if n < 3 {
        return (0.0, Vector2::ZERO);
    }
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        let cross = p.x * q.y - q.x * p.y;
        a2 += cross;
        cx += (p.x + q.x) * cross;
        cy += (p.y + q.y) * cross;
    }
    if a2.abs() < Real::EPSILON {
        return (0.0, Vector2::ZERO);
    }
    let area = a2 * 0.5;
    (area, snap_to_centre(Vector2::new(cx / (3.0 * a2), cy / (3.0 * a2)), area))
}

impl ExtrudedProfile {
    /// Area and centroid, **exactly**, arcs included.
    ///
    /// Exact and not tessellated because a centroid that is merely *nearly* right is not good
    /// enough: a rigid joint in Avian 0.7 diverges when a body's centre of mass is off by any
    /// amount at all — 1e-12 behaves exactly as 1e-2 does. A sampled boundary would put the
    /// centroid of a symmetric profile a few ulps off zero, which is the difference between a
    /// cantilever holding and reaching 1e17 metres.
    ///
    /// The region is a polygon through the boundary points plus, for each arc, the circular
    /// *segment* between its chord and itself. Both have closed forms.
    fn area_centroid(&self) -> (Real, Vector2) {
        if self.arcs.is_empty() {
            return polygon_area_centroid(&self.profile);
        }
        // Chord endpoints stand in for each arc, so the polygon is the straight-edged core.
        let mut core: Vec<Vector2> = self.profile.clone();
        for arc in &self.arcs {
            core.push(arc.at(arc.start));
            core.push(arc.at(arc.end));
        }
        let c = core.iter().fold(Vector2::ZERO, |a, b| a + *b) / core.len() as Real;
        core.sort_by(|a, b| {
            let (pa, pb) = ((a.y - c.y).atan2(a.x - c.x), (b.y - c.y).atan2(b.x - c.x));
            pa.partial_cmp(&pb).unwrap_or(core::cmp::Ordering::Equal)
        });
        core.dedup_by(|a, b| (*a - *b).length_squared() < Real::EPSILON);

        let (mut area, mut moment) = if core.len() >= 3 {
            let (a, c) = polygon_area_centroid(&core);
            (a, c * a)
        } else {
            (0.0, Vector2::ZERO)
        };

        for arc in &self.arcs {
            let theta = arc.sweep();
            let (r, s) = (arc.radius, theta.sin());
            // Circular segment: area r²(θ − sin θ)/2, centroid 4r·sin³(θ/2) / (3(θ − sin θ)) from
            // the circle's centre along the bisector. A full turn degenerates to the whole disc,
            // whose centroid is the centre — the formula's `sin θ = 0` handles it.
            let seg_area = 0.5 * r * r * (theta - s);
            if seg_area.abs() < Real::EPSILON {
                continue;
            }
            let bisector = arc.start + theta * 0.5;
            let d = if (theta - s).abs() < Real::EPSILON {
                0.0
            } else {
                4.0 * r * (theta * 0.5).sin().powi(3) / (3.0 * (theta - s))
            };
            let centroid = arc.center + Vector2::new(bisector.cos(), bisector.sin()) * d;
            area += seg_area;
            moment += centroid * seg_area;
        }
        if area.abs() < Real::EPSILON {
            return (0.0, Vector2::ZERO);
        }
        (area, snap_to_centre(moment / area, area))
    }

    /// Area, centroid, and the second moments about that centroid — Green's theorem over the
    /// boundary, which is exact for a polygon and is the same integral a section modulus needs.
    ///
    /// Returns `(area, centroid, ∫x²dA, ∫z²dA, ∫xz dA)`, the last three taken about the centroid.
    /// With arcs present the *moments* come from a sampled boundary (256 points per turn, so ~1e-5
    /// relative), while the area and centroid stay exact — see [`Self::area_centroid`] for why that
    /// asymmetry is deliberate.
    pub fn section(&self) -> (Real, Vector2, Real, Real, Real) {
        let boundary = self.tessellated();
        let (exact_area, exact_centroid) = self.area_centroid();
        let profile = &boundary;
        let n = profile.len();
        let (mut a2, mut cx, mut cz) = (0.0, 0.0, 0.0);
        let (mut ixx, mut izz, mut ixz) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let p = profile[i];
            let q = profile[(i + 1) % n];
            let cross = p.x * q.y - q.x * p.y;
            a2 += cross;
            cx += (p.x + q.x) * cross;
            cz += (p.y + q.y) * cross;
            ixx += (p.x * p.x + p.x * q.x + q.x * q.x) * cross;
            izz += (p.y * p.y + p.y * q.y + q.y * q.y) * cross;
            ixz += (p.x * q.y + 2.0 * p.x * p.y + 2.0 * q.x * q.y + q.x * p.y) * cross;
        }
        let _ = (cx, cz);
        let sampled_area = a2 * 0.5;
        if exact_area.abs() < Real::EPSILON || sampled_area.abs() < Real::EPSILON {
            return (0.0, Vector2::ZERO, 0.0, 0.0, 0.0);
        }
        // Moments about the origin, shifted to the centroid by the parallel-axis theorem, then
        // scaled by however much area the sampling lost — so they describe the exact region rather
        // than the inscribed polygon that stood in for it.
        let k = exact_area / sampled_area;
        let ixx = (ixx / 12.0 - sampled_area * exact_centroid.x * exact_centroid.x) * k;
        let izz = (izz / 12.0 - sampled_area * exact_centroid.y * exact_centroid.y) * k;
        let ixz = (ixz / 24.0 - sampled_area * exact_centroid.x * exact_centroid.y) * k;
        (exact_area, exact_centroid, ixx, izz, ixz)
    }
}

impl SupportMap for ExtrudedProfile {
    fn local_support_point(&self, dir: Vector) -> Vector {
        // The sweep is exact: the support in Y is always an end cap, whatever the profile does.
        let y = if dir.y >= 0.0 { self.half_depth } else { -self.half_depth };
        let d2 = Vector2::new(dir.x, dir.z);
        let mut best = Vector2::ZERO;
        let mut best_dot = Real::NEG_INFINITY;
        for p in &self.profile {
            let dot = p.x * d2.x + p.y * d2.y;
            if dot > best_dot {
                best_dot = dot;
                best = *p;
            }
        }
        // An arc supports every direction it spans, exactly: `centre + radius * dir`. This is the
        // whole point of carrying arcs — a drawn curve has a continuous normal instead of stepping
        // between facets. Outside its span the arc's endpoints, which are in `profile`, take over.
        if let Some(unit) = d2.try_normalize() {
            let angle = unit.y.atan2(unit.x);
            for arc in &self.arcs {
                if arc.spans(angle) {
                    let p = arc.center + unit * arc.radius;
                    let dot = p.x * d2.x + p.y * d2.y;
                    if dot > best_dot {
                        best_dot = dot;
                        best = p;
                    }
                }
            }
        } else if best_dot == Real::NEG_INFINITY {
            // Degenerate direction and no vertices: any point of the shape will do.
            if let Some(arc) = self.arcs.first() {
                best = arc.at(arc.start);
            }
        }
        Vector::new(best.x, y, best.y)
    }
}

impl PolygonalFeatureMap for ExtrudedProfile {
    fn local_support_feature(&self, dir: Vector, out: &mut PolygonalFeature) {
        let d2 = Vector2::new(dir.x, dir.z);

        // An arc facing this way has no flat side to clip against: the contact feature is the
        // *line* where the sweep touches it, exactly as `Cylinder` returns a segment on its curved
        // part. Two vertices, and the manifold code treats it as an edge.
        if let Some(unit) = d2.try_normalize() {
            let angle = unit.y.atan2(unit.x);
            if dir.y.abs() < d2.length() {
                for (a, arc) in self.arcs.iter().enumerate() {
                    if arc.spans(angle) {
                        let p = arc.center + unit * arc.radius;
                        out.vertices[0] = Vector::new(p.x, -self.half_depth, p.y);
                        out.vertices[1] = Vector::new(p.x, self.half_depth, p.y);
                        out.vids[0] = PackedFeatureId::vertex(1000 + a as u32 * 2);
                        out.vids[1] = PackedFeatureId::vertex(1001 + a as u32 * 2);
                        out.eids[0] = PackedFeatureId::edge(1000 + a as u32);
                        out.eids[1] = PackedFeatureId::edge(1000 + a as u32);
                        out.num_vertices = 2;
                        out.fid = PackedFeatureId::face(1000 + a as u32);
                        return;
                    }
                }
            }
        }

        // Arcs sampled, so a cap face is the whole boundary rather than just its corners.
        let boundary = self.tessellated();
        let profile: &[Vector2] = &boundary;
        let n = profile.len();
        if n == 0 {
            out.num_vertices = 0;
            return;
        }

        // Is the support face a cap, or a side? Compare how well the sweep axis matches `dir`
        // against the best the profile's own edge normals can do. No magic threshold: the two are
        // measured in the same units and the larger wins.
        let mut best_edge = 0usize;
        let mut best_edge_dot = Real::NEG_INFINITY;
        for i in 0..n {
            let p = profile[i];
            let q = profile[(i + 1) % n];
            // Outward normal of a counter-clockwise edge.
            let e = Vector2::new(q.x - p.x, q.y - p.y);
            let nrm = Vector2::new(e.y, -e.x);
            let len = (nrm.x * nrm.x + nrm.y * nrm.y).sqrt();
            if len <= Real::EPSILON {
                continue;
            }
            let dot = (nrm.x * d2.x + nrm.y * d2.y) / len;
            if dot > best_edge_dot {
                best_edge_dot = dot;
                best_edge = i;
            }
        }

        if dir.y.abs() >= best_edge_dot {
            // A cap. Four vertices *spread around* the profile rather than four consecutive ones,
            // which on a many-sided profile would be a sliver off to one side of the real contact
            // patch — the defect this shape's `ConvexPolyhedron` equivalent has.
            let cap = if dir.y >= 0.0 { self.half_depth } else { -self.half_depth };
            let count = n.min(4);
            let stride = (n / count.max(1)).max(1);
            for k in 0..count {
                // Reverse the winding on the -Y cap so the face stays outward-facing.
                let i = if dir.y >= 0.0 {
                    (k * stride) % n
                } else {
                    (n - (k * stride) % n) % n
                };
                let p = profile[i];
                out.vertices[k] = Vector::new(p.x, cap, p.y);
                out.vids[k] = PackedFeatureId::vertex(i as u32);
                out.eids[k] = PackedFeatureId::edge(i as u32);
            }
            out.num_vertices = count;
            out.fid = PackedFeatureId::face(if dir.y >= 0.0 { 0 } else { 1 });
        } else {
            // A side quad: one profile edge swept the full depth.
            let i = best_edge;
            let j = (i + 1) % n;
            let (p, q) = (profile[i], profile[j]);
            out.vertices[0] = Vector::new(p.x, -self.half_depth, p.y);
            out.vertices[1] = Vector::new(q.x, -self.half_depth, q.y);
            out.vertices[2] = Vector::new(q.x, self.half_depth, q.y);
            out.vertices[3] = Vector::new(p.x, self.half_depth, p.y);
            out.vids[0] = PackedFeatureId::vertex(i as u32);
            out.vids[1] = PackedFeatureId::vertex(j as u32);
            out.vids[2] = PackedFeatureId::vertex(j as u32 + n as u32);
            out.vids[3] = PackedFeatureId::vertex(i as u32 + n as u32);
            for k in 0..4 {
                out.eids[k] = PackedFeatureId::edge(i as u32 * 4 + k as u32);
            }
            out.num_vertices = 4;
            out.fid = PackedFeatureId::face(2 + i as u32);
        }
    }
}

// `Shape` requires both, and a support map is all either one needs — the same generic routines
// `Cylinder` and `Cone` use, rather than a hand-rolled projection that could disagree with the
// support function it is supposed to describe.
impl RayCast for ExtrudedProfile {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        local_ray_intersection_with_support_map_with_params(
            self,
            &mut VoronoiSimplex::new(),
            ray,
            max_time_of_impact,
            solid,
        )
    }
}

impl PointQuery for ExtrudedProfile {
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        local_point_projection_on_support_map(self, &mut VoronoiSimplex::new(), pt, solid)
    }

    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        (self.project_local_point(pt, false), FeatureId::Unknown)
    }
}

impl Shape for ExtrudedProfile {
    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        // A non-uniform scale in the profile plane turns a circle into an ellipse, which an arc
        // cannot represent — so say no rather than silently returning a different solid. Straight
        // edges scale exactly under any scale at all.
        if !self.arcs.is_empty() && (scale.x - scale.z).abs() > Real::EPSILON {
            return None;
        }
        let k = scale.x;
        Some(Box::new(Self {
            profile: self
                .profile
                .iter()
                .map(|p| Vector2::new(p.x * scale.x, p.y * scale.z))
                .collect(),
            arcs: self
                .arcs
                .iter()
                .map(|a| ProfileArc {
                    center: a.center * k,
                    radius: a.radius * k.abs(),
                    start: a.start,
                    end: a.end,
                })
                .collect(),
            half_depth: self.half_depth * scale.y.abs(),
        }))
    }

    fn compute_local_aabb(&self) -> Aabb {
        let mut lo = Vector2::splat(Real::INFINITY);
        let mut hi = Vector2::splat(Real::NEG_INFINITY);
        for p in &self.profile {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        // An arc reaches its circle's extreme wherever it spans an axis direction, and only its
        // endpoints otherwise — so ask the support function rather than bounding the whole circle.
        for dir in [Vector::X, Vector::NEG_X, Vector::Z, Vector::NEG_Z] {
            if !self.arcs.is_empty() {
                let p = self.local_support_point(dir);
                let p2 = Vector2::new(p.x, p.z);
                lo = lo.min(p2);
                hi = hi.max(p2);
            }
        }
        Aabb::new(
            Vector::new(lo.x, -self.half_depth, lo.y),
            Vector::new(hi.x, self.half_depth, hi.y),
        )
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        self.compute_local_aabb().bounding_sphere()
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        let (area, centroid, ixx, izz, ixz) = self.section();
        let depth = self.half_depth * 2.0;
        let mass = density * area.abs() * depth;
        if mass <= 0.0 {
            return MassProperties::new(Vector::ZERO, 0.0, Vector::ZERO);
        }
        // Extruding a section: the in-plane moments carry through the depth, and the depth itself
        // contributes a slab term to the two axes perpendicular to the sweep.
        let slab = area.abs() * depth * depth * depth / 12.0;
        let i_xx = density * (slab + depth * izz);
        let i_zz = density * (slab + depth * ixx);
        let i_yy = density * depth * (ixx + izz);
        // The product of inertia is generally non-zero for an arbitrary profile, so hand parry the
        // full tensor and let it find the principal frame rather than assuming one.
        let i_xz = -density * depth * ixz;
        let inertia = Matrix::from_cols(
            Vector::new(i_xx, 0.0, i_xz),
            Vector::new(0.0, i_yy, 0.0),
            Vector::new(i_xz, 0.0, i_zz),
        );
        MassProperties::with_inertia_matrix(
            Vector::new(centroid.x, 0.0, centroid.y),
            mass,
            inertia,
        )
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::ExtrudedProfile
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        // The thinnest the solid gets: the smaller of its in-plane extent and its depth.
        let aabb = self.compute_local_aabb();
        let e = aabb.extents();
        e.x.min(e.y).min(e.z)
    }

    fn ccd_angular_thickness(&self) -> Real {
        core::f32::consts::FRAC_PI_2 as Real
    }

    fn is_convex(&self) -> bool {
        true
    }

    fn as_support_map(&self) -> Option<&dyn SupportMap> {
        Some(self)
    }

    fn as_polygonal_feature_map(&self) -> Option<(&dyn PolygonalFeatureMap, Real)> {
        Some((self as &dyn PolygonalFeatureMap, 0.0))
    }

    fn feature_normal_at_point(&self, _feature: FeatureId, _point: Vector) -> Option<Vector> {
        None
    }

    fn compute_swept_aabb(&self, start: &Pose, end: &Pose) -> Aabb {
        let a = self.compute_aabb(start);
        let b = self.compute_aabb(end);
        a.merged(&b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{Cuboid, Cylinder};

    fn rect(hx: Real, hz: Real) -> ExtrudedProfile {
        ExtrudedProfile::new(
            alloc::vec![
                Vector2::new(-hx, -hz),
                Vector2::new(hx, -hz),
                Vector2::new(hx, hz),
                Vector2::new(-hx, hz),
            ],
            0.0,
        )
    }

    /// A rectangle extruded is a cuboid, so every number should match parry's own to rounding.
    #[test]
    fn a_rectangular_profile_is_a_cuboid() {
        let mut p = rect(0.3, 0.7);
        p.half_depth = 0.5;
        let c = Cuboid::new(Vector::new(0.3, 0.5, 0.7));

        for dir in [
            Vector::X,
            Vector::NEG_X,
            Vector::Y,
            Vector::NEG_Y,
            Vector::Z,
            Vector::new(1.0, 1.0, 1.0).normalize(),
            Vector::new(-2.0, 0.3, 1.0).normalize(),
        ] {
            let a = p.local_support_point(dir);
            let b = c.local_support_point(dir);
            // Compare what a support function actually promises — the *extent* along `dir` — not
            // the point. On a face-on direction a box has several equally-far vertices, and picking
            // a different one of them is not a disagreement.
            assert!(
                (a.dot(dir) - b.dot(dir)).abs() < 1.0e-5,
                "extent along {dir:?}: {} vs {}",
                a.dot(dir),
                b.dot(dir)
            );
            // And it must be a point *on* the box, not merely one that measures the same.
            assert!(
                a.x.abs() <= 0.3 + 1.0e-5 && a.y.abs() <= 0.5 + 1.0e-5 && a.z.abs() <= 0.7 + 1.0e-5,
                "support point outside the solid: {a:?}"
            );
        }

        let mp = p.mass_properties(2500.0);
        let mc = c.mass_properties(2500.0);
        assert!((mp.mass() - mc.mass()).abs() < 1.0e-3, "{} vs {}", mp.mass(), mc.mass());
        let (ip, ic) = (mp.principal_inertia(), mc.principal_inertia());
        // Sorted, because the two need not agree on which axis is listed first.
        let mut a = [ip.x, ip.y, ip.z];
        let mut b = [ic.x, ic.y, ic.z];
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1.0e-3, "inertia {a:?} vs {b:?}");
        }
    }

    /// Many-sided, it should converge on the cylinder it approximates — from *inside*, since an
    /// inscribed polygon is always a little smaller than its circle.
    #[test]
    fn many_sides_converge_on_a_cylinder() {
        let cyl = Cylinder::new(0.4, 0.25);
        for (sides, tol) in [(16usize, 0.03), (64, 0.002), (256, 0.0002)] {
            let p = ExtrudedProfile::regular(0.25, sides, 0.4);
            let m = p.mass_properties(1000.0).mass();
            let c = cyl.mass_properties(1000.0).mass();
            let deficit = (c - m) / c;
            assert!(deficit > 0.0, "{sides}-gon should be lighter than its circle");
            assert!(deficit < tol, "{sides}-gon deficit {deficit} exceeds {tol}");
        }
    }

    /// A single full arc **is** a cylinder — not an approximation of one.
    #[test]
    fn a_round_profile_is_a_cylinder() {
        let p = ExtrudedProfile::round(0.25, 0.4);
        let c = Cylinder::new(0.4, 0.25);

        for i in 0..64 {
            let a = i as Real / 64.0 * core::f32::consts::TAU as Real;
            for dy in [-2.0, -0.3, 0.0, 0.3, 2.0] {
                let dir = Vector::new(a.cos(), dy, a.sin());
                let (x, y) = (p.local_support_point(dir), c.local_support_point(dir));
                assert!(
                    (x.dot(dir) - y.dot(dir)).abs() < 1.0e-5,
                    "extent along {dir:?}: {} vs {}",
                    x.dot(dir),
                    y.dot(dir)
                );
            }
        }

        let (mp, mc) = (p.mass_properties(1000.0), c.mass_properties(1000.0));
        assert!((mp.mass() - mc.mass()).abs() / mc.mass() < 1.0e-4, "{} vs {}", mp.mass(), mc.mass());
    }

    /// **The centroid of a centred profile must be exactly zero, not nearly zero.**
    ///
    /// Avian 0.7 diverges when a body in a `FixedJoint` has a centre of mass off by *any* amount —
    /// 1e-12 behaves exactly as 1e-2 does. So the area and centroid are computed in closed form
    /// (polygon plus circular segments) rather than from the sampled boundary the inertia uses.
    #[test]
    fn a_centred_profile_has_an_exactly_zero_centroid() {
        let cases = [
            ExtrudedProfile::round(0.25, 0.4),
            ExtrudedProfile::regular(0.3, 16, 0.2),
            ExtrudedProfile::new(
                alloc::vec![
                    Vector2::new(-0.4, -0.2),
                    Vector2::new(0.4, -0.2),
                    Vector2::new(0.4, 0.2),
                    Vector2::new(-0.4, 0.2),
                ],
                0.5,
            ),
        ];
        for p in cases {
            let com = p.mass_properties(2500.0).local_com;
            assert_eq!(com, Vector::ZERO, "centre of mass is {com:?}, not exactly zero");
        }
    }

    /// An arc's side feature is a *segment* along the sweep, the way a cylinder's curved part is —
    /// there is no flat face there to clip against.
    #[test]
    fn an_arc_side_feature_is_a_segment() {
        let p = ExtrudedProfile::round(0.3, 0.5);
        let mut f = PolygonalFeature::default();
        p.local_support_feature(Vector::X, &mut f);
        assert_eq!(f.num_vertices, 2, "a curved side has no polygon face");
        for i in 0..2 {
            let v = f.vertices[i];
            assert!((Vector2::new(v.x, v.z).length() - 0.3).abs() < 1.0e-5);
            assert!((v.y.abs() - 0.5).abs() < 1.0e-5);
        }
    }

    /// The support feature has to be the *whole* face, not a sliver of it.
    #[test]
    fn the_cap_feature_spans_the_profile() {
        let p = ExtrudedProfile::regular(0.3, 16, 0.6);
        let mut f = PolygonalFeature::default();
        p.local_support_feature(Vector::Y, &mut f);
        assert_eq!(f.num_vertices, 4);
        let v: Vec<Vector2> = (0..4)
            .map(|i| Vector2::new(f.vertices[i].x, f.vertices[i].z))
            .collect();
        // Area of the quad against the cap's own; four spread vertices give the inscribed square,
        // 2/pi of the circle. Four *consecutive* ones would give a few percent.
        let quad = 0.5 * ((v[2] - v[0]).perp_dot(v[3] - v[1])).abs();
        let cap = core::f32::consts::PI as Real * 0.3 * 0.3;
        assert!(quad / cap > 0.5, "cap feature covers only {:.1}%", quad / cap * 100.0);
    }

    /// A side face is a quad of the swept edge — all four corners on the surface.
    #[test]
    fn the_side_feature_is_the_swept_edge() {
        let p = ExtrudedProfile::regular(0.3, 8, 0.6);
        let mut f = PolygonalFeature::default();
        p.local_support_feature(Vector::X, &mut f);
        assert_eq!(f.num_vertices, 4);
        for i in 0..4 {
            let v = f.vertices[i];
            assert!((v.y.abs() - 0.6).abs() < 1.0e-5, "corner off the caps: {v:?}");
            let r = Vector2::new(v.x, v.z).length();
            assert!(r <= 0.3 + 1.0e-5, "corner outside the profile: {r}");
        }
    }
}
