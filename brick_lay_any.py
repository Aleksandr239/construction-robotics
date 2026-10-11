# On Play, read the brick meshes in the open stage and lay the ones the Panda can reach.
# A brick past the reach is left on the pallet. That skip does not stop the run.
# Bricks are separate meshes about 20 x 10 x 8 cm. The pallet cubes stay 8 cm along X.

import traceback

import numpy as np
from omni.behavior.scripting.core import BehaviorScript

ROOT = (
    "c:/users/alkis/downloads/isaac-sim-standalone-6.0.1-windows-x86_64/"
    "extsDeprecated/isaacsim.robot_motion.motion_generation/"
)
SAFE = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=np.float64)
ANCHOR = np.array([-1.01, 1.37, -2.82, -2.42, 0.36, 1.09, -1.55], dtype=np.float64)
LO = np.array([-2.8973, -1.7638, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HI = np.array([2.8973, 1.7638, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Hand local Y closes on world +X, local Z points world down.
R_GRASP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
OPEN = 7.0
CLOSED = 5.4
GRASP_DZ = 15.5


class BrickLayAny(BehaviorScript):
    def on_init(self):
        self._clear()

    def on_play(self):
        self._clear()
        self.phase = "boot"
        self.note = "starting"
        try:
            import carb
            carb.settings.get_settings().set("/app/scripting/ignoreWarningDialog", True)
        except Exception:
            pass
        self._mark()

    def on_pause(self):
        return

    def on_stop(self):
        self.phase = "hold"

    def on_update(self, current_time, delta_time):
        try:
            self._tick()
        except Exception:
            self.note = "stopped: " + traceback.format_exc().splitlines()[-1]
            self.phase = "hold"
            self._mark()
            self._hold()

    def _clear(self):
        self.robot = None
        self.bricks = None
        self.pv = None
        self.ik = None
        self.idx = np.array([0], dtype=np.int32)
        self.jaw_l = None
        self.jaw_r = None
        self.boot = 0
        self.intro = 0
        self.cursor = 0
        self.phase = "boot"
        self.q = SAFE.copy()
        self.finger = OPEN
        self.last_arm = SAFE.copy()
        self.last_finger = OPEN
        self.which = -1
        self.offset = None
        self.note = "ready"
        self.base = np.array([10.0, 438.0, 0.0])
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.fwd = np.array([-1.0, 0.0])
        self.perp = np.array([0.0, 1.0])
        self.slots = np.zeros((0, 3))
        self.place_at = np.zeros((0, 3))
        self.place_q = np.zeros((0, 4))
        self.frames_q = np.zeros((0, 7))
        self.frames_f = np.zeros((0,))
        self.frames_c = np.zeros((0,), dtype=np.int32)
        self.frames_s = np.zeros((0,), dtype=np.int32)
        self.n_plan = 0
        self.n_skip = 0
        self.n_snap = 0
        self.planned = False

    def _mark(self):
        prim = self.prim
        if prim is None or not prim.IsValid():
            return
        attr = prim.GetAttribute("brickLay:note")
        if not attr:
            from pxr import Sdf
            attr = prim.CreateAttribute("brickLay:note", Sdf.ValueTypeNames.String)
        attr.Set(str(self.note))

    def _send(self, arm, finger):
        full = np.zeros((1, 9), dtype=np.float32)
        full[0, :7] = np.asarray(arm, dtype=np.float32)
        full[0, 7:] = float(finger)
        self.pv.set_dof_positions(full, self.idx)
        self.pv.set_dof_position_targets(full, self.idx)
        self.pv.set_dof_velocities(np.zeros((1, 9), dtype=np.float32), self.idx)
        self.last_arm = np.array(arm, dtype=np.float64, copy=True)
        self.last_finger = float(finger)
        self.q = self.last_arm.copy()
        self.finger = float(finger)

    def _hold(self):
        if self.pv is None or self.last_arm is None:
            return
        if not np.all(np.isfinite(self.last_arm)):
            return
        try:
            self._send(self.last_arm, self.last_finger)
            self._jaws(self.last_arm, self.last_finger)
        except Exception:
            pass

    def _ensure(self):
        if self.robot is not None and self.pv is not None and self.ik is not None:
            return True
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

        robot = SingleArticulation("/World/Franka")
        robot.initialize()
        pv = robot._articulation_view._physics_view
        lim = pv.get_dof_limits().copy()
        lim[0, 7, 0] = 0.0
        lim[0, 7, 1] = 8.0
        lim[0, 8, 0] = 0.0
        lim[0, 8, 1] = 8.0
        pv.set_dof_limits(lim, self.idx)
        self._read_base()
        ik = LulaKinematicsSolver(
            ROOT + "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml",
            ROOT + "motion_policy_configs/franka/lula_franka_gen.urdf",
        )
        ik.set_robot_base_pose(self.base, self.base_quat)
        self.robot = robot
        self.pv = pv
        self.ik = ik
        self._ensure_jaws()
        self._send(SAFE, OPEN)
        return True

    def _read_base(self):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath("/World/Franka")
        ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        if t is not None:
            self.base = np.array([float(t[0]), float(t[1]), float(t[2])])
        o = ops.get("xformOp:orient")
        if o is not None:
            w = float(o.GetReal())
            im = o.GetImaginary()
            q = np.array([w, float(im[0]), float(im[1]), float(im[2])])
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
            if abs(q[0]) < 1e-2 and abs(q[3]) > 0.9:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            self.base_quat = q

    def _ensure_jaws(self):
        from pxr import Gf, Sdf, UsdGeom
        stage = self.stage
        if not stage.GetPrimAtPath("/World/GripperJaws"):
            UsdGeom.Xform.Define(stage, "/World/GripperJaws")
        for name, attr_name in (("Left", "jaw_l"), ("Right", "jaw_r")):
            path = "/World/GripperJaws/" + name
            prim = stage.GetPrimAtPath(path)
            if not prim:
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
                xf.AddScaleOp().Set(Gf.Vec3d(0.9, 3.2, 9.2))
                cube.GetPrim().CreateAttribute(
                    "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
                ).Set([Gf.Vec3f(0.16, 0.16, 0.17)])
                prim = cube.GetPrim()
            setattr(self, attr_name, prim.GetAttribute("xformOp:translate"))

    def _tick(self):
        if self.phase == "hold":
            self._hold()
            return
        if self.phase == "boot":
            self.boot += 1
            if self.boot < 12:
                return
            if not self._ensure():
                if self.boot > 50:
                    self.phase = "hold"
                    self.note = "robot did not start"
                    self._mark()
                return
            self._send(SAFE, OPEN)
            if not self.planned:
                self.note = "reading the wall"
                self._mark()
                self._build()
                self.planned = True
            self.phase = "intro"
            self.intro = 0
            self.note = "laying 0/%d, skipped %d" % (self.n_plan, self.n_skip)
            self._mark()
            return
        if self.robot is None:
            self._ensure()
        if self.phase == "intro":
            t = min(1.0, self.intro / 36.0)
            arm = (1.0 - t) * SAFE + t * ANCHOR
            self._send(arm, OPEN)
            self._jaws(arm, OPEN)
            self.intro += 1
            if t >= 1.0:
                self.phase = "lay" if len(self.frames_q) else "hold"
                if self.phase == "hold":
                    self.note = "nothing in reach, skipped %d" % self.n_skip
                    self._mark()
            return
        if self.cursor >= len(self.frames_q):
            self.phase = "hold"
            self.note = "done, placed %d, skipped %d" % (self.n_snap, self.n_skip)
            self._mark()
            self._hold()
            return
        arm = self.frames_q[self.cursor]
        finger = float(self.frames_f[self.cursor])
        carry = int(self.frames_c[self.cursor])
        snap = int(self.frames_s[self.cursor])
        self._send(arm, finger)
        self._jaws(arm, finger)
        if carry >= 0:
            if self.which != carry:
                self.offset = self.slots[carry] - self._finger_mid(arm)
                self.which = carry
            self._park(carry, self._finger_mid(arm) + self.offset, np.array([1.0, 0.0, 0.0, 0.0]))
        if snap >= 0:
            self._park(snap, self.place_at[snap], self.place_q[snap])
            if self.which == snap:
                self.which = -1
                self.offset = None
            self.n_snap = max(self.n_snap, int(np.sum(np.unique(self.frames_s[: self.cursor + 1]) >= 0)))
        self.cursor += 1
        if self.cursor % 24 == 0:
            shown = len(set(int(i) for i in self.frames_s[: self.cursor] if i >= 0))
            self.note = "laying %d/%d, skipped %d" % (shown, self.n_plan, self.n_skip)
            self._mark()

    def _build(self):
        targets = self._scan_targets()
        self._aim(targets)
        self._prepare_supply(len(targets))
        if self.bricks is None or len(self.slots) == 0 or len(targets) == 0:
            self.n_plan = 0
            self.n_skip = len(targets)
            self._log("no work targets=%d slots=%d" % (len(targets), len(self.slots)))
            return
        bands = []
        for z in sorted(float(t["center"][2]) for t in targets):
            if not bands or z - bands[-1] > 6.0:
                bands.append(z)

        def course_of(z):
            return min(range(len(bands)), key=lambda i: abs(float(z) - bands[i]))

        base_xy = self.base[:2]
        avail = sorted(
            range(len(self.slots)),
            key=lambda i: (
                (round(float(self.slots[i][0])) - base_xy[0]) ** 2
                + (round(float(self.slots[i][1])) - base_xy[1]) ** 2,
                -float(self.slots[i][2]),
            ),
        )
        frames = []
        planned = 0
        skipped = 0
        placed = set()
        q = ANCHOR.copy()

        def supports(ti):
            # Bricks of the course below that this one rests on.
            target = targets[ti]
            course = course_of(target["center"][2])
            if course == 0:
                return []
            span = float(max(target["size"][0], target["size"][1])) * 0.9
            out = []
            for other, below in enumerate(targets):
                if course_of(below["center"][2]) != course - 1:
                    continue
                if abs(float(below["center"][0]) - float(target["center"][0])) > 6.0:
                    continue
                if abs(float(below["center"][1]) - float(target["center"][1])) < span:
                    out.append(other)
            return out

        def attempt(ti):
            nonlocal planned, skipped, q
            target = targets[ti]
            place = target["center"] + np.array([0.0, 0.0, GRASP_DZ])
            shoulder = self.base + np.array([0.0, 0.0, 33.0])
            if float(np.linalg.norm(place - shoulder)) > 108.0:
                skipped += 1
                return "far"
            # A brick resting on a gap would topple and wreck the course below.
            if any(s not in placed for s in supports(ti)):
                skipped += 1
                return "far"
            fails = 0
            for k, bi in enumerate(list(avail)):
                start_q = q if k == 0 else ANCHOR.copy()
                brick_frames, q_end, reason = self._plan_one(start_q, bi, target, place)
                if brick_frames is None:
                    q = ANCHOR.copy()
                    if reason == "grasp":
                        avail.remove(bi)
                        continue
                    fails += 1
                    if fails >= 4:
                        return "later"
                    continue
                avail.remove(bi)
                frames.extend(brick_frames)
                frames.extend(self._lerp(q_end, ANCHOR, 28, OPEN))
                q = ANCHOR.copy()
                planned += 1
                placed.add(ti)
                self.place_at[bi] = target["center"]
                self.place_q[bi] = target["quat"]
                return "placed"
            skipped += 1
            return "far"

        groups = {}
        for index, target in enumerate(targets):
            groups.setdefault(course_of(target["center"][2]), []).append(index)
        for course in sorted(groups):
            row = sorted(
                groups[course],
                key=lambda i: (targets[i]["center"][1], targets[i]["center"][0]),
            )
            later = []
            for ti in row:
                if attempt(ti) == "later":
                    later.append(ti)
            for ti in later:
                if attempt(ti) == "later":
                    skipped += 1
                    q = ANCHOR.copy()
        self.n_plan = planned
        self.n_skip = skipped
        if frames:
            self.frames_q = np.array([f[0] for f in frames], dtype=np.float32)
            self.frames_f = np.array([f[1] for f in frames], dtype=np.float32)
            self.frames_c = np.array([f[2] for f in frames], dtype=np.int32)
            self.frames_s = np.array([f[3] for f in frames], dtype=np.int32)
        self._log("planned %d skipped %d frames %d targets %d" % (planned, self.n_skip, len(frames), len(targets)))

    def _plan_one(self, q, brick_index, target, place):
        slot = self.slots[brick_index]
        grasp = slot + np.array([0.0, 0.0, GRASP_DZ])
        hover = grasp + np.array([0.0, 0.0, 10.0])
        bag = []
        seg, q = self._track(q, hover, False)
        if seg is None:
            seg, q = self._track(ANCHOR.copy(), grasp, True)
            if seg is None:
                return None, ANCHOR.copy(), "grasp"
            bag.extend((qq, OPEN, -1, -1) for qq in seg)
        else:
            bag.extend((qq, OPEN, -1, -1) for qq in seg)
            seg, q = self._track(q, grasp, True)
            if seg is None:
                return None, ANCHOR.copy(), "grasp"
            bag.extend((qq, OPEN, -1, -1) for qq in seg)
        hold = q.copy()
        for i in range(1, 17):
            finger = OPEN + (CLOSED - OPEN) * (i / 16.0)
            carry = brick_index if i == 16 else -1
            bag.append((hold.copy(), finger, carry, -1))
        carried = None
        for route in self._routes(slot, target["center"], place):
            q = hold.copy()
            steps = []
            for i, goal in enumerate(route):
                seg, q = self._track(q, goal, i == len(route) - 1)
                if seg is None:
                    steps = None
                    break
                steps.extend((qq, CLOSED, brick_index, -1) for qq in seg)
            if steps is not None:
                carried = steps
                break
        if carried is None:
            return None, ANCHOR.copy(), "travel"
        bag.extend(carried)
        hold = q.copy()
        for i in range(1, 13):
            finger = CLOSED + (OPEN - CLOSED) * (i / 12.0)
            bag.append((hold.copy(), finger, -1, brick_index))
        return bag, q, "ok"

    def _routes(self, slot, target, place):
        # Over the top first for the low courses, straight in front first for the high ones,
        # and the other way round as a fallback.
        over = self._route_over(slot, target, place)
        front = self._route_front(slot, target, place)
        return [over, front]

    def _lead(self, slot):
        base = self.base
        side = float(np.dot(np.asarray(slot[:2], dtype=np.float64) - base[:2], self.perp))
        return np.array([*(base[:2] - self.fwd * 12.0 + self.perp * side), float(base[2]) + 60.0])

    def _route_over(self, slot, target, place):
        # Carry the brick along a corridor above the course, then lower it straight down.
        fwd = self.fwd
        perp = self.perp
        base = self.base
        z0 = float(base[2])
        v1 = np.array([*(base[:2] - fwd * 10.0 + perp * -42.0), z0 + 62.0])
        gate = np.array([*(base[:2] + fwd * 15.0 + perp * -32.0), z0 + 58.0])
        x_safe = float(base[0] + fwd[0] * 18.0)
        z_safe = max(56.0, min(float(place[2]) + 18.0, 70.0))
        y_slot = float(place[1])
        ys = []
        yy = float(gate[1])
        while abs(yy - y_slot) > 10.0:
            yy = yy - 10.0 if y_slot < yy else yy + 10.0
            ys.append(yy)
        ys.append(y_slot)
        pts = [self._lead(slot), v1, gate]
        for yy in ys:
            pts.append(np.array([x_safe, float(yy), z_safe]))
        front_z = min(max(float(place[2]) + 22.0, z0 + 50.0), z_safe + 4.0)
        pts.append(np.array([float(base[0] + fwd[0] * 40.0), y_slot, front_z]))
        pts.append(np.array([float(place[0]), y_slot, min(float(place[2]) + 18.0, z_safe)]))
        pts.append(place)
        return pts

    def _route_front(self, slot, target, place):
        fwd = self.fwd
        perp = self.perp
        base = self.base
        z0 = float(base[2])
        lead = self._lead(slot)
        a = float(np.dot(target[:2] - base[:2], perp))
        if abs(a) < 30.0:
            s = 1.0 if a >= 0.0 else -1.0
            v1 = np.array([*(base[:2] - fwd * 10.0 + perp * (s * 42.0)), z0 + 62.0])
            v2 = np.array([*(base[:2] + fwd * 15.0 + perp * (s * 32.0)), z0 + 55.0])
            xy = target[:2] - fwd * 12.0
            f = float(np.dot(xy - base[:2], fwd))
            v3xy = base[:2] + fwd * f + perp * (s * 14.0)
            v3 = np.array([v3xy[0], v3xy[1], max(place[2] + 18.0, z0 + 40.0)])
            return [lead, v1, v2, v3, place]
        v1 = np.array([*(base[:2] - fwd * 10.0 + perp * a), z0 + 60.0])
        v2 = np.array([*(base[:2] + fwd * 10.0 + perp * a), z0 + 50.0])
        v3 = np.array([*(base[:2] + fwd * 10.0 + perp * a), max(place[2] + 14.0, z0 + 36.0)])
        return [lead, v1, v2, v3, place]
    def _track(self, q, goal, tight):
        tol = 1.2 if tight else 2.5
        omax = 0.25 if tight else 0.6
        seg = []
        best = 1e9
        stall = 0
        err = 1e9
        for _ in range(110):
            q, err, ori = self._step(q, goal)
            if not np.all(np.isfinite(q)):
                return None, ANCHOR.copy()
            seg.append(q.copy())
            if err < best - 0.15:
                best = err
                stall = 0
            else:
                stall += 1
            if err < tol and ori < omax:
                return seg, q
            if stall > 36 and err > tol:
                return None, q
        if err < tol:
            return seg, q
        return None, q

    def _step(self, q, target):
        p, rot = self._fk(q)
        jac = np.zeros((6, 7))
        eps = 1e-3
        for i in range(7):
            d = np.zeros(7)
            d[i] = eps
            p2, r2 = self._fk(q + d)
            jac[:3, i] = (p2 - p) / eps
            jac[3:, i] = self._vee(r2 @ rot.T) / eps
        ep = np.asarray(target, dtype=np.float64) - p
        eo = self._vee(R_GRASP @ rot.T)
        err = np.concatenate([ep, eo * 20.0])
        j6 = jac.copy()
        j6[3:] *= 20.0
        dq = j6.T @ np.linalg.solve(j6 @ j6.T + 6.0 * np.eye(6), err)
        n = float(np.max(np.abs(dq)))
        lim = 0.08 if float(np.linalg.norm(ep)) > 8.0 else 0.045
        if n > lim:
            dq *= lim / n
        q2 = np.clip(q + dq, LO + 0.02, HI - 0.02)
        return q2, float(np.linalg.norm(ep)), float(np.linalg.norm(eo))

    def _fk(self, q):
        pos, rot = self.ik.compute_forward_kinematics("panda_hand", np.array(q, dtype=np.float64))
        return np.array(pos, dtype=np.float64).reshape(3), np.array(rot, dtype=np.float64).reshape(3, 3)

    def _vee(self, m):
        return np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) * 0.5

    def _lerp(self, q0, q1, n, finger):
        out = []
        for i in range(1, n + 1):
            t = i / float(n)
            out.append(((1.0 - t) * q0 + t * q1, finger, -1, -1))
        return out

    def _ghost(self, prim):
        from pxr import UsdPhysics, UsdShade
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            else:
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
            mat = self.stage.GetPrimAtPath("/World/Looks/PhantomBrick")
            if mat and mat.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat))
        except Exception:
            pass

    def _scan_targets(self):
        from pxr import Usd, UsdGeom
        found = []
        stage = self.stage
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if path.startswith("/World/Franka") or "/BrickSupply" in path or "/Gripper" in path or "/Ground" in path:
                continue
            if path.startswith("/World/Brick_wall"):
                self._ghost(prim)
            bound = UsdGeom.Mesh(prim).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            rng = bound.ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            if any(not np.isfinite(mn[i]) or not np.isfinite(mx[i]) for i in range(3)):
                continue
            size = np.array([float(mx[i] - mn[i]) for i in range(3)])
            if float(np.min(size)) <= 0:
                continue
            if np.any(np.abs(np.sort(size) - np.array([8.0, 10.0, 20.0])) > np.array([3.0, 3.5, 4.0])):
                continue
            center = np.array([(float(mn[i]) + float(mx[i])) * 0.5 for i in range(3)])
            found.append({"center": center, "size": size, "quat": self._brick_quat(size), "path": path})
        return found
    def _brick_quat(self, size):
        long_axis = int(np.argmax(size[:2]))
        if long_axis == 0:
            return np.array([0.70710678, 0.0, 0.0, 0.70710678])
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _aim(self, targets):
        if not targets:
            self.fwd = np.array([-1.0, 0.0])
            self.perp = np.array([0.0, 1.0])
            return
        mu = np.mean([t["center"][:2] for t in targets], axis=0)
        fwd = mu - self.base[:2]
        if abs(fwd[0]) >= abs(fwd[1]):
            fwd = np.array([1.0 if fwd[0] >= 0 else -1.0, 0.0])
        else:
            fwd = np.array([0.0, 1.0 if fwd[1] >= 0 else -1.0])
        self.fwd = fwd
        self.perp = np.array([-fwd[1], fwd[0]])

    def _prepare_supply(self, n_targets):
        from isaacsim.core.prims import RigidPrim
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = self.stage
        root = stage.GetPrimAtPath("/World/BrickSupply")
        if not root:
            UsdGeom.Xform.Define(stage, "/World/BrickSupply")
            root = stage.GetPrimAtPath("/World/BrickSupply")
        have = list(root.GetChildren())
        if n_targets > len(have) and have:
            self._spawn_more(n_targets - len(have), have[0])
            have = list(root.GetChildren())
        have = sorted(have, key=lambda prim: prim.GetName())
        slots = []
        for prim in have:
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            slots.append([float(t[0]), float(t[1]), float(t[2])])
        if not slots:
            self.slots = np.zeros((0, 3))
            return
        bricks = RigidPrim(prim_paths_expr="/World/BrickSupply/Brick_.*")
        bricks.initialize()
        try:
            paths = [str(path) for path in bricks.prim_paths]
            by_path = {str(prim.GetPath()): row for prim, row in zip(have, slots)}
            ordered = [by_path[path] for path in paths]
            if len(ordered) == len(paths):
                slots = ordered
        except Exception:
            pass
        self.slots = np.array(slots, dtype=np.float64)
        n = len(self.slots)
        self.place_at = self.slots.copy()
        self.place_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        ident = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
        bricks.set_world_poses(self.slots.astype(np.float32), ident)
        bricks.set_linear_velocities(np.zeros((n, 3), dtype=np.float32))
        bricks.set_angular_velocities(np.zeros((n, 3), dtype=np.float32))
        self.bricks = bricks

    def _spawn_more(self, count, proto):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        stage = self.stage
        root = "/World/BrickSupply"
        existing = []
        for prim in stage.GetPrimAtPath(root).GetChildren():
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            t = m.ExtractTranslation()
            existing.append((round(float(t[0]), 1), round(float(t[1]), 1), round(float(t[2]), 1)))
        binding = proto.GetRelationship("material:binding")
        bound = list(binding.GetTargets()) if binding and binding.GetTargets() else []
        made = 0
        names = {p.GetName() for p in stage.GetPrimAtPath(root).GetChildren()}
        n = 0
        for z in (9.0, 19.0, 29.0, 39.0, 49.0):
            for y in (378.0, 400.0, 422.0, 444.0, 466.0):
                for x in (28.0, 44.0, 60.0, 76.0):
                    if (x, y, z) in existing:
                        continue
                    if (x - self.base[0]) ** 2 + (y - self.base[1]) ** 2 < 400.0:
                        continue
                    while True:
                        name = "Brick_%02d" % n
                        n += 1
                        if name not in names:
                            break
                    path = root + "/" + name
                    cube = UsdGeom.Cube.Define(stage, path)
                    cube.CreateSizeAttr(1.0)
                    xf = UsdGeom.Xformable(cube.GetPrim())
                    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
                    xf.AddScaleOp().Set(Gf.Vec3d(8.0, 20.0, 10.0))
                    cube.GetPrim().CreateAttribute(
                        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
                    ).Set([Gf.Vec3f(0.55, 0.22, 0.16)])
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2.0)
                    if bound:
                        cube.GetPrim().CreateRelationship("material:binding").SetTargets(bound)
                    names.add(name)
                    made += 1
                    if made >= count:
                        return

    def _finger_mid(self, q):
        left, _ = self.ik.compute_forward_kinematics("panda_leftfinger", np.array(q, dtype=np.float64))
        right, _ = self.ik.compute_forward_kinematics("panda_rightfinger", np.array(q, dtype=np.float64))
        return (np.array(left, dtype=np.float64) + np.array(right, dtype=np.float64)) * 0.5

    def _jaws(self, q, finger):
        if not self.jaw_l or not self.jaw_r:
            return
        from pxr import Gf
        pos, rot = self.ik.compute_forward_kinematics("panda_hand", np.array(q, dtype=np.float64))
        base = np.array(pos, dtype=np.float64).reshape(3)
        matrix = np.array(rot, dtype=np.float64).reshape(3, 3)
        bite = float(finger) - 1.15
        left = base + matrix @ np.array([0.0, bite, 10.44])
        right = base + matrix @ np.array([0.0, -bite, 10.44])
        self.jaw_l.Set(Gf.Vec3d(float(left[0]), float(left[1]), float(left[2])))
        self.jaw_r.Set(Gf.Vec3d(float(right[0]), float(right[1]), float(right[2])))

    def _park(self, index, pos, quat):
        self.bricks.set_world_poses(
            np.array([pos], dtype=np.float32),
            np.array([quat], dtype=np.float32),
            indices=np.array([index], dtype=np.int32),
        )
        self.bricks.set_linear_velocities(
            np.zeros((1, 3), dtype=np.float32),
            indices=np.array([index], dtype=np.int32),
        )
        self.bricks.set_angular_velocities(
            np.zeros((1, 3), dtype=np.float32),
            indices=np.array([index], dtype=np.int32),
        )

    def _log(self, text):
        try:
            path = r"C:\Users\alkis\Desktop\Robots\isaac_scene_mcp\scripts\brick_lay_any_log.txt"
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except Exception:
            pass
