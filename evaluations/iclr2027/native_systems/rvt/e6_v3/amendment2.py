"""Amendment 2: only the native Open Drawer missed-interaction close boundary."""
from evaluations.iclr2027.native6_v3.open_drawer_missed_interaction import OpenDrawerMissedInteraction
from evaluations.iclr2027.native6_v3.events import FaultDecision


class DrawerContactOnsetMixin:
    def _init_amendment2(self):
        scope = (self.row['task'] == 'open_drawer' and self.row['fault_family'] == 'missed_interaction'
                 and all('maintained_contact' in s['relation_sources'] for s in self.root_specs.values()))
        self.drawer_onset = OpenDrawerMissedInteraction() if scope and not self.passive else None
        self.drawer_pending = None
        self.drawer_closing = False

    def _observe_amendment2(self, sample):
        state = getattr(self, 'drawer_onset', None)
        if state is None or self.injecting:
            return
        decision = state.observe_completed_step(**self.stamp(),
            actual_open_amount=min(sample['actual_open_amount']),
            native_closure_in_progress=self.drawer_closing,
            validated_contact_objects=sample['contact_objects'], contact_source='simGetContactInfo')
        if decision is not None:
            self.drawer_pending = decision
            self.transactions.append(dict(kind='amendment_2_contact_onset', **self.stamp(),
                decision=decision.to_dict(), physical_state=sample, contact_source='simGetContactInfo'))

    def _drawer_close_transaction(self, scene, action, sample):
        state = self.drawer_onset
        closing = all(x > .9 for x in sample['actual_open_amount']) and float(action[0]) <= .5
        if state.intervention_started or not closing:
            return self.original_gripper_action(scene, action)
        started = state.begin_close_transaction(**self.stamp(), arm='single',
            arm_motion_completed=True, actual_open_amount=min(sample['actual_open_amount']),
            native_open_to_close_started=closing, active_drawer_objects=list(self.roots))
        if not started:
            return self.original_gripper_action(scene, action)
        self.transactions.append(dict(kind='amendment_2_close_begin', **self.stamp(),
            physical_state=sample, requested_open=float(action[0]), arm_motion_completed=True))
        self.drawer_closing = True
        attach = self.mode._attach_grasped_objects
        try:
            # Native control law and velocity are unchanged. The completed-step
            # wrapper can reverse this transaction before the next closing step.
            return self.original_gripper_action(scene, action)
        finally:
            self.drawer_closing = False
            self.mode._attach_grasped_objects = attach
            if not state.intervention_started:
                state.finish_without_contact()

    def _reverse_drawer_at_onset(self):
        decision = self.drawer_pending
        self.drawer_pending = None
        state = self.drawer_onset
        self.injecting = True
        self.drawer_closing = False
        state.mark_reverse_open_started(decision, **self.stamp(), applied_open_amount=1.)
        common = FaultDecision(**decision.to_dict(), eligible=True)
        self.state.eligible = True
        self.state.eligible_event = common
        self.state.mark_injected(common)
        self.force_open_transaction = True
        self.mode._attach_grasped_objects = False
        self.transactions.append(dict(kind='amendment_2_reverse_open_begin', **self.stamp(),
            applied_open_amount=1., physical_state=self.sample()))
        # Count forbidden *calls*, not physical displacement caused by dynamics.
        counts = dict(drawer_setter_calls=0, arm_target_changes=0)
        patched = []
        def guard(obj, name, counter):
            if not hasattr(obj, name):
                return
            original = getattr(obj, name)
            def forbidden(*args, **kwargs):
                counts[counter] += 1
                raise RuntimeError('Amendment 2 forbids '+name+' during reverse-open')
            patched.append((obj, name, original))
            setattr(obj, name, forbidden)
        try:
            for obj in self.roots.values():
                for name in ('set_pose', 'set_position', 'set_orientation', 'set_quaternion'):
                    guard(obj, name, 'drawer_setter_calls')
            for obj in self.objects.values():
                if obj.get_type().name.lower() == 'joint' and 'drawer' in obj.get_name().lower():
                    for name in ('set_joint_position', 'set_joint_target_position', 'set_joint_target_velocity'):
                        guard(obj, name, 'drawer_setter_calls')
            for name in ('set_joint_positions', 'set_joint_target_positions', 'set_joint_target_velocities'):
                guard(self.arm, name, 'arm_target_changes')
            # PyRep's convenience actuator retains the previous velocity sign.
            # A deliberate reversal otherwise looks like oscillation and returns
            # done immediately. Begin a new actuator transaction by resetting
            # only its Python bookkeeping, never simulator joint state/targets.
            self.gripper._prev_positions = [None] * self.gripper._num_joints
            self.gripper._prev_vels = [None] * self.gripper._num_joints
            self._open()
            sample = self.sample()
            evidence = dict(mode='maintained_contact_onset_reverse_open', **counts,
                intervention_occurrences=state.intervention_occurrences, physical_state=sample)
            clear = (all(x > .9 for x in sample['actual_open_amount'])
                     and not set(self.roots).intersection(sample['contact_objects']+sample['attachment_objects']))
            if clear:
                state.confirm_physical_effect(**self.stamp(), actual_open_amount=min(sample['actual_open_amount']),
                    validated_contact_objects=sample['contact_objects'], attachment_objects=sample['attachment_objects'], **counts)
                self._confirm(evidence)
            self.transactions.append(dict(kind='amendment_2_reverse_open_end', **self.stamp(),
                physical_effect_confirmed=state.physical_effect_confirmed, evidence=evidence))
        finally:
            for obj, name, original in reversed(patched):
                setattr(obj, name, original)
            self.injecting = False
