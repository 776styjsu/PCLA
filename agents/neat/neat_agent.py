import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image, ImageDraw

from leaderboardcodes import autonomous_agent1
from neat.architectures import AttentionField
from neat.config import GlobalConfig
from neat.utils import flow_to_color
from planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'MultiTaskAgent'


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (image.width // scale, image.height // scale)
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


class MultiTaskAgent(autonomous_agent1.AutonomousAgent):

	def setup(self, path_to_conf_file):
		self.track = autonomous_agent1.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
		self.args = json.load(args_file)
		args_file.close()
		self.args['out_res'] = 100
		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque()}

		self.config = GlobalConfig()
		self.net = AttentionField(self.config, 'cuda')
		self.net.encoder.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_encoder.pth')))
		self.net.decoder.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_decoder.pth')))

		self.plan_grid = self.net.create_plan_grid(self.config.plan_scale, self.config.plan_points, 1)
		self.light_grid = self.net.create_light_grid(self.config.light_x_steps, self.config.light_y_steps, 1)

		self.net.cuda()
		self.net.eval()

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'bev').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'flow').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'out').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'img').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'meta').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'sal').mkdir(parents=True, exist_ok=False)

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_left'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_right'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov': 100,
					'id': 'rgb_front'
					},
				{	
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z': 25,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 800, 'height': 800, 'fov': 100,
					'id': 'bev'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		result = {
				'rgb': rgb,
				'rgb_left': rgb_left,
				'rgb_right': rgb_right,
				'rgb_front': rgb_front,
				'bev': bev,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}

		pos = self._get_position(result)
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	# @torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		tick_data = self.tick(input_data)

		if self.step < self.config.seq_len:
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']))).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']))).unsqueeze(0)
			self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
			rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']))).unsqueeze(0)
			self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)
		target_point = torch.stack(tick_data['target_point']).to('cuda', dtype=torch.float32)
		self.target_point_model = target_point

		rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']))).unsqueeze(0)
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
		
		rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']))).unsqueeze(0)
		self.input_buffer['rgb_left'].popleft()
		self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
		
		rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']))).unsqueeze(0)
		self.input_buffer['rgb_right'].popleft()
		self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))
		
		images = []
		for i in range(self.config.seq_len):
			images.append(self.input_buffer['rgb'][i])
			if self.config.num_camera == 3:
				images.append(self.input_buffer['rgb_left'][i])
				images.append(self.input_buffer['rgb_right'][i])

		# encoding = self.net.encoder(images, gt_velocity)
		
		# pred_waypoint_mean, red_light_occ = self.net.plan(target_point, encoding, self.plan_grid, self.light_grid, self.config.plan_points, self.config.plan_iters)
		# steer, throttle, brake, metadata = self.net.control_pid(pred_waypoint_mean[:, self.config.seq_len:], gt_velocity, target_point, red_light_occ)
		with torch.no_grad():
			encoding = self.net.encoder(images, gt_velocity)
			pred_waypoint_mean, red_light_occ = self.net.plan(
				target_point, encoding, self.plan_grid, self.light_grid,
				self.config.plan_points, self.config.plan_iters
			)
			steer, throttle, brake, metadata = self.net.control_pid(
				pred_waypoint_mean[:, self.config.seq_len:], gt_velocity, target_point, red_light_occ
			)


		self.encoding_model = encoding
		self.pred_waypoint_mean_model = pred_waypoint_mean
		self.pid_metadata = metadata

		steer = float(steer)
		throttle = float(throttle)
		brake = float(brake)

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0
		
		control = carla.VehicleControl()
		control.steer = steer
		control.throttle = throttle
		control.brake = brake

		# === SALIENCY START (fresh graph; grad only on latest frame) ===
		with torch.enable_grad():
			cams = ['rgb', 'rgb_left', 'rgb_right'] if self.config.num_camera == 3 else ['rgb']
			last_t = self.config.seq_len - 1

			# Rebuild inputs the encoder expects (seq_len * num_cams),
			# enabling grads ONLY on the last timestep frames.
			grad_images = []
			for i in range(self.config.seq_len):
				for cam in cams:
					x = self.input_buffer[cam][i].detach().clone()
					x.requires_grad_(i == last_t)
					grad_images.append(x)

			# Detach non-visual conditions so they don't carry graph
			tp = target_point.detach()
			vel = gt_velocity.detach()

			enc_g = self.net.encoder(grad_images, vel)
			pred_wp_g, red_occ_g = self.net.plan(
				tp, enc_g, self.plan_grid, self.light_grid,
				self.config.plan_points, self.config.plan_iters
			)

			# Explain the x-offset of the first FUTURE waypoint
			scalar = pred_wp_g[:, self.config.seq_len, 0].sum()

			# Collect grads w.r.t. the grad-enabled inputs (last timestep across cams)
			base = last_t * len(cams)
			grad_vars = [grad_images[base + k] for k in range(len(cams))]

			grads = torch.autograd.grad(
				scalar, grad_vars,
				retain_graph=False, create_graph=False, allow_unused=True
			)

			saliency_maps = {}
			for cam, g in zip(cams, grads):
				if g is None:
					continue
				sal = g.abs().sum(dim=1, keepdim=True)              # (1,1,H,W)
				sal = (sal / (sal.max() + 1e-8)).detach().cpu().squeeze(0).squeeze(0).numpy()
				saliency_maps[cam] = sal

			self.saliency_maps = saliency_maps
		# === SALIENCY END ===

		# Print out some control information
		tp0 = tick_data['target_point'][0].item() if torch.is_tensor(tick_data['target_point'][0]) else tick_data['target_point'][0]
		tp1 = tick_data['target_point'][1].item() if torch.is_tensor(tick_data['target_point'][1]) else tick_data['target_point'][1]
		spd = float(tick_data['speed'])

		print(
			f'step={self.step}, command={int(tick_data["next_command"])}, '
			f'throttle={throttle:.2f}, brake={brake:.2f}, steer={steer:.2f}, '
			f'speed={spd:.2f}, target=({tp0:.2f},{tp1:.2f})'
		)

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)

		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# grid used for visualizing occupancy and flow
		linspace_x = torch.linspace(-self.config.axis/2, self.config.axis/2, steps=self.args['out_res'])
		linspace_y = torch.linspace(-self.config.axis/2, self.config.axis/2, steps=self.args['out_res'])
		linspace_t = torch.linspace(0, self.config.tot_len - 1, steps=self.config.tot_len)

		for i in range(self.config.seq_len):
			# save one sample per batch
			front_numpy = (self.input_buffer['rgb'][i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
			left_numpy = (self.input_buffer['rgb_left'][i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
			right_numpy = (self.input_buffer['rgb_right'][i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
			image_numpy = np.concatenate([left_numpy,front_numpy,right_numpy], axis=1)
			image_display = Image.fromarray(image_numpy)
			if not os.path.isdir(self.save_path / 'img' / str(frame).zfill(4)):
				os.mkdir(self.save_path / 'img' / str(frame).zfill(4))
			image_display.save(f"{self.save_path}/img/{str(frame).zfill(4)}/{str(i)}.png")

		# target point in pixel coordinates
		target_point_pixel = self.target_point_model.squeeze().cpu().numpy()
		target_point_pixel[1] += self.config.offset * self.config.resolution

		# hack for when actual target is outside image (axis/2 * resolution)
		target_point_pixel = np.clip(target_point_pixel, -(self.config.axis/2 * self.config.resolution - 1), (self.config.axis/2 * self.config.resolution - 1)) 
		target_point_pixel = (target_point_pixel*self.args['out_res']//50 + self.args['out_res']//2).astype(np.uint8)
		
		for i in range(self.config.tot_len):
			# predicted waypoint in pixel coordinates
			pred_waypoint = self.pred_waypoint_mean_model[0,i].data.cpu().numpy()
			pred_waypoint[1] += self.config.offset * self.config.resolution
			pred_waypoint = np.clip(pred_waypoint, -(self.config.axis/2 * self.config.resolution - 1), (self.config.axis/2 * self.config.resolution - 1)) 
			pred_waypoint = (pred_waypoint*self.args['out_res']//(self.config.axis * self.config.resolution) + self.args['out_res']//2).astype(np.uint8)
			
			# visualization of occupancy and flow
			img_rows = []
			flow_rows = []
			for row in range(self.args['out_res']):
				grid_x, grid_y, grid_t = torch.meshgrid(linspace_x, linspace_y[row], linspace_t[i].unsqueeze(0))
				grid_points = torch.stack((grid_x, grid_y, grid_t), dim=3).unsqueeze(0).repeat(1,1,1,1,1)
				grid_points = grid_points.reshape(1,-1,3).to('cuda', dtype=torch.float32)
				pred_img_pts, pred_img_offsets, _ = self.net.decode(grid_points, self.target_point_model, self.encoding_model)
				pred_img_pts = torch.argmax(pred_img_pts[-1], dim=1)
				pred_img = pred_img_pts.reshape(1,self.args['out_res'])
				pred_flow = pred_img_offsets[-1].reshape(1,2,self.args['out_res'])
				img_rows.append(pred_img)
				flow_rows.append(pred_flow)
			
			pred_img = torch.stack(img_rows, dim=-1)
			pred_flow = torch.stack(flow_rows, dim=-1)

			semantics = pred_img[0,:,:].transpose(1, 0).data.cpu().numpy().astype(np.uint8)
			semantic_display = np.zeros((semantics.shape[0], semantics.shape[1], 3))
			for key, value in self.config.classes.items():
				semantic_display[np.where(semantics == key)] = value
			semantic_display = semantic_display.astype(np.uint8)
			semantic_display = Image.fromarray(semantic_display)
			if not os.path.isdir(self.save_path / 'out' / str(frame).zfill(4)):
				os.mkdir(self.save_path / 'out' / str(frame).zfill(4))
			semantic_display.save(f"{self.save_path}/out/{str(frame).zfill(4)}/{str(i)}.png")

			# flow image of predicted offsets
			flow_uv = pred_flow[0,:,:,:].transpose(2,0).data.cpu().numpy()*self.args['out_res']/self.config.axis
			flow_rgb = flow_to_color(flow_uv)

			flow_display = Image.fromarray(flow_rgb)
			draw = ImageDraw.Draw(flow_display)
			draw.ellipse([tuple(target_point_pixel-1), tuple(target_point_pixel+1)], fill='Blue', outline='Blue')
			draw.ellipse([tuple(pred_waypoint-1), tuple(pred_waypoint+1)], fill='Red', outline='Red')
			if not os.path.isdir(self.save_path / 'flow' / str(frame).zfill(4)):
				os.mkdir(self.save_path / 'flow' / str(frame).zfill(4))
			flow_display.save(f"{self.save_path}/flow/{str(frame).zfill(4)}/{str(i)}.png")

		# === SAVE SALIENCY ONCE PER FRAME ===
		if hasattr(self, 'saliency_maps') and isinstance(self.saliency_maps, dict):
			frame_dir = self.save_path / 'sal' / str(frame).zfill(4)
			frame_dir_raw = self.save_path / 'sal_raw' / str(frame).zfill(4)
			frame_dir.mkdir(parents=True, exist_ok=True)
			frame_dir_raw.mkdir(parents=True, exist_ok=True)

			cams = ['rgb', 'rgb_left', 'rgb_right'] if self.config.num_camera == 3 else ['rgb']
			last_t = self.config.seq_len - 1

			for cam in cams:
				sal = self.saliency_maps.get(cam)
				if sal is None:
					continue

				# normalize to [0,1] and build heatmap
				sal = np.clip(sal, 0.0, 1.0)
				sal_u8 = (sal * 255).astype(np.uint8)  # (H, W)
				heat_bgr = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)

				# fetch the explained input frame (256x256 crop from buffer)
				img_t = (self.input_buffer[cam][last_t][0]
							.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8))  # RGB
				img_bgr = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR)

				overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heat_bgr, 0.4, 0.0)
				overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

				Image.fromarray(overlay_rgb).save(frame_dir / f'{cam}.png')
				Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)).save(frame_dir_raw / f'{cam}_heat.png')
				Image.fromarray(sal_u8).save(frame_dir_raw / f'{cam}_gray.png')

	def destroy(self):
		del self.net


