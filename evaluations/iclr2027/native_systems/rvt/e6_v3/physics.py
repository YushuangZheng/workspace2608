"""Episode-local v3 instrumentation; no policy input or shared-file mutation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from evaluations.iclr2027.native6_v3.events import EventGroundedFaultState, quaternion_angle_xyzw
from evaluations.iclr2027.native6_v3.physical_clock import SimulatorStepClock
from evaluations.iclr2027.native6_v3.object_mapping import active_roots, load_mapping, normalize_coppelia_name
from .mapping.candidate_filter import contact_candidate, attachment_candidate
from .amendment2 import DrawerContactOnsetMixin

BASE = Path(__file__).resolve().parents[3]

def describe(obj):
    chain = []; seen = {obj.get_handle()}; parent = obj.get_parent()
    while parent is not None:
        if parent.get_handle() in seen:
            raise RuntimeError('cyclic scene ancestry')
        seen.add(parent.get_handle()); chain.append(parent.get_name()); parent = parent.get_parent()
    return dict(name=obj.get_name(), handle=obj.get_handle(),
                object_type=obj.get_type().name.lower(), parent_chain=chain)

class PhysicalAdapter(DrawerContactOnsetMixin):
    def __init__(self, backend, row, *, passive=False):
        from pyrep.backend import sim
        self.sim = sim; self.backend = backend; self.row = row
        self.scene = backend.raw._scene; self.gripper = self.scene.robot.gripper
        self.arm = self.scene.robot.arm; self.pyrep = self.scene.pyrep
        self.mapping = load_mapping(BASE/'configs/shared/native6_object_contact_mapping_v3.json')
        self.protocol = json.loads((BASE/'configs/shared/native6_physical_protocol_v3.json').read_text())
        self.passive = passive
        e = self.protocol['eligibility']
        self.state = EventGroundedFaultState(row['fault_family'],
            translation_threshold_m=e['motion_translation_m'], rotation_threshold_rad=e['motion_rotation_rad'],
            stable_relation_seconds=e['stable_relation_seconds'])
        self.objects = {o.get_handle(): o for o in self.pyrep.get_objects_in_tree()}
        self.roots = {}; self.root_specs = {}
        for spec in active_roots(self.mapping, row['task'], row['variation']):
            matches = [o for o in self.objects.values() if normalize_coppelia_name(o.get_name()) == spec['scene_root_name']]
            if len(matches) != 1 or matches[0].get_type().name.lower() != 'shape':
                raise RuntimeError('frozen semantic shape did not resolve uniquely: '+spec['scene_root_name'])
            self.roots[spec['canonical_id']] = matches[0]; self.root_specs[spec['canonical_id']] = spec
        self.fingers = []
        for name in self.mapping['gripper_contact_scope']['allowed_collision_shape_base_names']:
            matches = [o for o in self.objects.values() if normalize_coppelia_name(o.get_name()) == name]
            if len(matches) != 1 or matches[0].get_type().name.lower() != 'shape':
                raise RuntimeError('frozen finger shape did not resolve uniquely: '+name)
            self.fingers.append(matches[0])
        self.start_time = float(sim.simGetSimulationTime())
        self.events = []; self.transactions = []; self.pending = None
        self.injecting = False; self.force_open_transaction = False
        self._init_amendment2()
        self.onset = None; self.end = None; self.restored = None
        self.relation_candidate_since = None
        self.mode = backend.raw._action_mode.gripper_action_mode
        self.original_gripper_action = self.mode.action
        self.original_actuate = self.gripper.actuate
        self.clock = SimulatorStepClock(self.pyrep, simulation_time=sim.simGetSimulationTime,
            simulation_timestep=self.pyrep.get_simulation_timestep, on_completed_step=self._observe_step).install()
        self.observed_step = self.pyrep.step
        self.pyrep.step = self._step
        self.mode.action = self._gripper_action
        self.gripper.actuate = self._actuate

    def time(self):
        return max(0., float(self.sim.simGetSimulationTime()) - self.start_time)

    def stamp(self):
        return dict(sim_step=self.clock.completed_steps, simulation_time_s=self.time())

    def sample(self):
        attachments = []
        for obj in self.gripper.get_grasped_objects():
            name = attachment_candidate(self.mapping, self.row['task'], self.row['variation'], describe(obj))
            if name is not None: attachments.append(name)
        contacts = []; pairs = []
        for finger in self.fingers:
            for info in self.sim.simGetContactInfo(finger.get_handle(), True):
                handles = [int(h) for h in info['contact_handles']]
                if finger.get_handle() not in handles: raise RuntimeError('contact query returned unrelated pair')
                other = handles[1] if handles[0] == finger.get_handle() else handles[0]
                if other not in self.objects: raise RuntimeError('contact object absent from scene inventory')
                obj = self.objects[other]
                name = contact_candidate(self.mapping, self.row['task'], self.row['variation'], describe(obj))
                if name is not None:
                    contacts.append(name)
                    pairs.append(dict(finger_name=finger.get_name(), task_shape_name=obj.get_name(),
                        contact_handles=handles, canonical_id=name, contact_info=[float(x) for x in info['contact']]))
        maintained = [n for n in contacts if 'maintained_contact' in self.root_specs[n]['relation_sources']]
        source = 'attachment' if attachments else ('maintained_contact' if maintained else None)
        names = attachments if attachments else maintained
        return dict(**self.stamp(), event_kind='completed_physics_state', arm='single',
            object_names=sorted(set(names)), actual_open_amount=[float(x) for x in self.gripper.get_open_amount()],
            attachment_objects=sorted(set(attachments)), contact_objects=sorted(set(contacts)),
            relation_source=source, contact_pairs=pairs)

    def _observe_step(self, record):
        # This callback is strictly read-only with respect to the simulator.
        sample = self.sample(); self.events.append(sample)
        if self.passive: return
        self._observe_amendment2(sample)
        if not self.injecting:
            decision = self.state.observe_relation(**self.stamp(), arm='single',
                source=sample['relation_source'], object_names=sample['object_names'], contact_evidence='sim_contact_info')
            if decision is not None: self.pending = decision
        # Restoration is reported only after the identical target relation has
        # physically re-established for the public stability duration.
        event = self.state.trigger_event
        if self.onset is not None and event and event.object_names and self.restored is None:
            target = set(event.object_names)
            observed = set(sample['attachment_objects'] if self.row['task'] != 'open_drawer' else sample['contact_objects'])
            if target <= observed:
                if self.relation_candidate_since is None: self.relation_candidate_since = self.time()
                if self.time() - self.relation_candidate_since + 1e-12 >= self.protocol['eligibility']['stable_relation_seconds']:
                    self.end = self.stamp(); self.restored = self.stamp()
            else: self.relation_candidate_since = None

    def _step(self, *args, **kwargs):
        value = self.observed_step(*args, **kwargs)
        if getattr(self, 'drawer_pending', None) is not None and not self.injecting:
            self._reverse_drawer_at_onset()
        # Injection is OUTSIDE the observational clock callback. Nested steps
        # during physical opening are observed, but cannot inject recursively.
        if self.pending is not None and not self.injecting:
            decision = self.pending; self.pending = None
            self._relation_loss(decision)
        return value

    def _actuate(self, amount, velocity):
        return self.original_actuate(1. if self.force_open_transaction else amount, velocity)

    def _confirm(self, evidence):
        if not self.state.physical_effect_confirmed:
            self.state.confirm_physical_effect(**self.stamp(), evidence=evidence)
            self.onset = self.stamp()

    def _open(self):
        self.gripper.release()
        started = self.time()
        while True:
            done = self.original_actuate(1., .2)
            self.pyrep.step(); self.scene.task.step()
            if done: break
            if self.time() - started > 10.: raise RuntimeError('gripper opening failed to terminate in 10 simulator seconds')

    def _relation_loss(self, decision):
        self.injecting = True
        try:
            self.state.mark_injected(decision)
            evidence = dict(source=decision.interaction_source)
            if decision.interaction_source == 'attachment':
                targets = {n:self.roots[n] for n in decision.object_names}
                positions = {n:np.asarray(o.get_position()).copy() for n,o in targets.items()}
                self.gripper.release()
                offset = np.asarray(self.protocol['medium']['attachment_relation_loss_translation_world_xyz'])
                for n,obj in targets.items(): obj.set_position((positions[n]+offset).tolist())
                displacement = {n:float(np.linalg.norm(np.asarray(o.get_position())-positions[n])) for n,o in targets.items()}
                sample = self.sample()
                evidence.update(displacement_m=displacement, attachment_objects=sample['attachment_objects'])
                if not set(targets)&set(sample['attachment_objects']) and all(v>=.035 for v in displacement.values()): self._confirm(evidence)
            else:
                self.force_open_transaction = True
                self._open()
                sample = self.sample(); evidence.update(actual_open_amount=sample['actual_open_amount'], contact_objects=sample['contact_objects'])
                if all(x>.9 for x in sample['actual_open_amount']) and not set(decision.object_names)&set(sample['contact_objects']): self._confirm(evidence)
            self.transactions.append(dict(kind='relation_loss', decision=decision.to_dict(), evidence=evidence))
        finally: self.injecting = False

    def _gripper_action(self, scene, action):
        if self.passive: return self.original_gripper_action(scene, action)
        if self.force_open_transaction:
            self.transactions.append(dict(kind='relation_loss_remaining_gripper_transaction', applied_open=1.))
            return self.original_gripper_action(scene, np.asarray([1.]))
        sample = self.sample()
        if getattr(self, 'drawer_onset', None) is not None:
            return self._drawer_close_transaction(scene, action, sample)
        closing = all(x>.9 for x in sample['actual_open_amount']) and float(action[0])<=.5
        if self.row['task']=='open_drawer': targets=sample['contact_objects']
        else:
            sensor = self.gripper._proximity_sensor
            targets = [n for n,o in self.roots.items() if 'attachment' in self.root_specs[n]['relation_sources'] and sensor.is_detected(o)]
        decision = self.state.observe_close_attempt(**self.stamp(), arm='single',
            actual_transition_to_closed=closing, detected_task_objects=targets)
        if decision is None: return self.original_gripper_action(scene, action)
        self.state.mark_injected(decision)
        if self.row['task']=='open_drawer':
            value=self.original_gripper_action(scene,np.asarray([1.]))
            sample=self.sample()
            evidence=dict(mode='maintained_contact_close_suppression', actual_open_amount=sample['actual_open_amount'],
                contact_objects=sample['contact_objects'], attachment_objects=sample['attachment_objects'])
            if all(x>.9 for x in sample['actual_open_amount']) and not set(targets)&set(sample['contact_objects']+sample['attachment_objects']): self._confirm(evidence)
        else:
            poses={n:self.roots[n].get_pose().copy() for n in targets}
            old=self.mode._attach_grasped_objects; self.mode._attach_grasped_objects=False
            try: value=self.original_gripper_action(scene,action)
            finally: self.mode._attach_grasped_objects=old
            attached=set(self.sample()['attachment_objects'])
            for n,pose in poses.items():
                if n not in attached: self.roots[n].set_pose(pose)
            errors={n:float(np.linalg.norm(np.asarray(self.roots[n].get_position())-np.asarray(pose[:3]))) for n,pose in poses.items()}
            evidence=dict(mode='attachment_suppression', restoration_errors_m=errors, attachment_objects=sorted(attached))
            if not set(targets)&attached and all(v<=.001 for v in errors.values()): self._confirm(evidence)
        self.transactions.append(dict(kind='missed_interaction', decision=decision.to_dict(), evidence=evidence))
        return value

    def step(self, command):
        self.force_open_transaction=False
        try:
            if not self.passive:
                pose=np.asarray(self.arm.get_tip().get_pose())
                decision=self.state.observe_motion_command(**self.stamp(),arm='single',
                    translation_m=float(np.linalg.norm(np.asarray(command[:3])-pose[:3])),
                    rotation_rad=quaternion_angle_xyzw(command[3:7],pose[3:7]))
                if decision is not None:
                    self.state.mark_injected(decision)
                    positions=self.arm.get_joint_positions(); self.arm.set_joint_target_positions(positions)
                    self.gripper.set_joint_target_velocities([0.]*len(self.gripper.joints))
                    begin=self.time(); translations=[]; rotations=[]
                    while self.time()-begin+1e-12<self.protocol['medium']['actuation_delay_seconds']:
                        self.pyrep.step(); self.scene.task.step()
                        actual=np.asarray(self.arm.get_tip().get_pose())
                        translations.append(float(np.linalg.norm(actual[:3]-pose[:3])))
                        rotations.append(quaternion_angle_xyzw(actual[3:7],pose[3:7]))
                    evidence=dict(hold_seconds=self.time()-begin,maximum_tcp_translation_m=max(translations),maximum_tcp_rotation_rad=max(rotations))
                    if evidence['hold_seconds']+1e-12>=.6 and max(translations)<=.001 and max(rotations)<=np.deg2rad(1.):
                        self._confirm(evidence); self.end=self.stamp()
                    self.transactions.append(dict(kind='actuation_delay_hold',joint_target_positions=positions,evidence=evidence))
                    success,terminal=self.scene.task.success()
                    return self.backend.raw.get_observation(),float(success),bool(terminal)
            return self.backend.raw.step(command)
        finally:
            self.force_open_transaction=False
            self.clock.raise_if_failed()

    def summary(self):
        value=self.state.summary()
        result={k:value[k] for k in ('eligible','injection_triggered','physical_effect_confirmed')}
        result['physically_triggered']=result['physical_effect_confirmed']
        for prefix,key in [('eligible','eligible_event'),('trigger','trigger_event'),('effect','effect_event')]:
            event=value[key]
            result[prefix+'_sim_step']=event['sim_step'] if event else None
            result[prefix+'_time_s']=event['simulation_time_s'] if event else None
        for prefix,event in [('violation_onset',self.onset),('violation_end',self.end),('relation_restored',self.restored)]:
            result[prefix+'_sim_step']=event['sim_step'] if event else None
            result[prefix+'_time_s']=event['simulation_time_s'] if event else None
        result.update(completed_sim_steps=self.clock.completed_steps,final_simulation_time_s=self.time())
        return result

    def close(self):
        self.mode.action=self.original_gripper_action
        self.gripper.actuate=self.original_actuate
        self.pyrep.step=self.observed_step
        self.clock.uninstall()
