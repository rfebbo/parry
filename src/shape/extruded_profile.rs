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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtrudedProfile {
    /// Convex boundary loop in the XZ plane, counter-clockwise.
    pub profile: Vec<Vector2>,
    /// Half the extrusion length, along Y.
    pub half_depth: Real,
}

impl ExtrudedProfile {
    /// A profile (XZ, convex, counter-clockwise) swept `half_depth * 2` along Y.
    pub fn new(profile: Vec<Vector2>, half_depth: Real) -> Self {
        Self { profile, half_depth }
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

    /// Area, centroid, and the second moments about that centroid — Green's theorem over the
    /// boundary, which is exact for a polygon and is the same integral a section modulus needs.
    ///
    /// Returns `(area, centroid, ∫x²dA, ∫z²dA, ∫xz dA)`, the last three taken about the centroid.
    pub fn section(&self) -> (Real, Vector2, Real, Real, Real) {
        let n = self.profile.len();
        let (mut a2, mut cx, mut cz) = (0.0, 0.0, 0.0);
        let (mut ixx, mut izz, mut ixz) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let p = self.profile[i];
            let q = self.profile[(i + 1) % n];
            let cross = p.x * q.y - q.x * p.y;
            a2 += cross;
            cx += (p.x + q.x) * cross;
            cz += (p.y + q.y) * cross;
            ixx += (p.x * p.x + p.x * q.x + q.x * q.x) * cross;
            izz += (p.y * p.y + p.y * q.y + q.y * q.y) * cross;
            ixz += (p.x * q.y + 2.0 * p.x * p.y + 2.0 * q.x * q.y + q.x * p.y) * cross;
        }
        let area = a2 * 0.5;
        if area.abs() < Real::EPSILON {
            return (0.0, Vector2::ZERO, 0.0, 0.0, 0.0);
        }
        let centroid = Vector2::new(cx / (3.0 * a2), cz / (3.0 * a2));
        // About the origin, then shifted to the centroid by the parallel-axis theorem.
        let ixx = ixx / 12.0 - area * centroid.x * centroid.x;
        let izz = izz / 12.0 - area * centroid.y * centroid.y;
        let ixz = ixz / 24.0 - area * centroid.x * centroid.y;
        (area, centroid, ixx, izz, ixz)
    }
}

impl SupportMap for ExtrudedProfile {
    fn local_support_point(&self, dir: Vector) -> Vector {
        // The sweep is exact: the support in Y is always an end cap, whatever the profile does.
        let y = if dir.y >= 0.0 { self.half_depth } else { -self.half_depth };
        let d2 = Vector2::new(dir.x, dir.z);
        let mut best = self.profile[0];
        let mut best_dot = Real::NEG_INFINITY;
        for p in &self.profile {
            let dot = p.x * d2.x + p.y * d2.y;
            if dot > best_dot {
                best_dot = dot;
                best = *p;
            }
        }
        Vector::new(best.x, y, best.y)
    }
}

impl PolygonalFeatureMap for ExtrudedProfile {
    fn local_support_feature(&self, dir: Vector, out: &mut PolygonalFeature) {
        let n = self.profile.len();
        let d2 = Vector2::new(dir.x, dir.z);

        // Is the support face a cap, or a side? Compare how well the sweep axis matches `dir`
        // against the best the profile's own edge normals can do. No magic threshold: the two are
        // measured in the same units and the larger wins.
        let mut best_edge = 0usize;
        let mut best_edge_dot = Real::NEG_INFINITY;
        for i in 0..n {
            let p = self.profile[i];
            let q = self.profile[(i + 1) % n];
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
                let p = self.profile[i];
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
            let (p, q) = (self.profile[i], self.profile[j]);
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
        // Exact for any scale: a swept convex profile stays a swept convex profile.
        Some(Box::new(Self {
            profile: self
                .profile
                .iter()
                .map(|p| Vector2::new(p.x * scale.x, p.y * scale.z))
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
