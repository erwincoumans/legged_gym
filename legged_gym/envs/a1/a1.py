# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs

from time import time
import numpy as np
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)


import os
import nvtx
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

use_cuda_interop = True

  
############################
#use pip install pytinydiffsim to get pytinyopengl3

import math, time
RGB_DATA=1
DEPTH_DATA=2
SEGMENTATION_DATA=3



############################

class A1(LeggedRobot):
  

    def create_sim(self):
      
      ############################
        #use pip install pytinydiffsim to get pytinyopengl3
        
        self._camera_width = 120
        self._camera_height = 80
        
        
        
        self.tiled_egl = True
        if self.tiled_egl:
        
          import pytinyopengl3 as g
          import math, time
          RGB_DATA=1
          DEPTH_DATA=2
          SEGMENTATION_DATA=3
          print("self.num_envs=",self.num_envs)
          self.max_x = math.ceil(math.sqrt(self.num_envs))
          width = self.max_x * self._camera_width
          height = self.max_x * self._camera_height
          self.width = width
          self.height = height
          print("opengl width=", width)
          print("opengl height=", height)
  
  
          self.viz = g.OpenGLUrdfVisualizer(width=width, height=height, window_type=2)#, window_type=0, render_device=-1)
          self.viz.opengl_app.set_background_color(0.3,0.3,0.3)
          self.viz.opengl_app.swap_buffer()
          self.viz.opengl_app.swap_buffer()


        # We need to create the OpenGL context before calling create_sim, otherwise it fails (Torch?)
        
        super().create_sim()
          
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        print("asset_root=",asset_root)
        print("asset_file=",asset_file)
        
        
        
        
        # Setup Tiled EGL renderer
        if self.tiled_egl:
          
          
          self.show_data = RGB_DATA #DEPTH_DATA #RGB_DATA #SEGMENTATION_DATA
          
          if self.matplotlib:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()
            img = np.random.rand(400, 400)
            
            if self.show_data==DEPTH_DATA:
              self.matplot_image = plt.imshow(img, interpolation='none', cmap='gray', vmin=0.8, vmax=1)
            else:
              self.matplot_image = plt.imshow(img, interpolation='none')
              
            ax = plt.gca()
          

          
          self.frame = 0
          
          
          # use window_type=2 for EGL and pass in render_device=0,1,... to select a specific GPU (or -1 for first available GPU)
          self.viz.opengl_app.set_background_color(1,1,1)#0.3,0.3,0.3)
          self.viz.opengl_app.swap_buffer()
          self.viz.opengl_app.swap_buffer()
          
          if use_cuda_interop:
              self.render_texid = self.viz.opengl_app.enable_render_to_texture(self.width, self.height)
              self.viz.opengl_app.swap_buffer()
              self.viz.opengl_app.swap_buffer()
              self.cuda_tex = self.viz.opengl_app.cuda_register_texture_image(self.render_texid, True)
              self.cuda_num_bytes = self.width*self.height*4*2 #4 component half-float, each half-float 2 bytes
              print("cuda_num_bytes=", self.cuda_num_bytes)
              self.ttensor = torch.zeros(self.width*self.height*4, dtype=torch.float16, device="cuda")
              self.cuda_mem = self.ttensor.data_ptr()

          
          # enable the following line (dump_frames_to_video) to create a video, requires ffmpeg installed
          # self.viz.opengl_app.dump_frames_to_video("video.mp4")
          
          urdf = g.OpenGLUrdfStructures()
          parser = g.UrdfParser()
          file_name = asset_root + "/a1_obj.urdf"
          print("robot file_name=",file_name)
          urdf = parser.load_urdf(file_name)
          print("urdf=",urdf)
          texture_path = "trunk_A1.png"
          self.viz.path_prefix = g.extract_path(file_name)
          print("self.viz.path_prefix=",self.viz.path_prefix)
          self.viz.convert_visuals(urdf, texture_path)
          print("create_instances")
      
          #all_instances_prev = self.viz.create_instances(urdf, texture_path, self.num_envs)
          all_instances_org = self.viz.create_instances(urdf, texture_path, self.num_envs)
          self.all_instances = []
          for env in all_instances_org:
            pairs = []
            for p in env:
              pair = g.UrdfInstancePair()
              pair.link_index = p.link_index
              pair.visual_instance = p.visual_instance
              pair.viz_origin_xyz = p.viz_origin_xyz
              pair.viz_origin_rpy = p.viz_origin_rpy
              pairs.append(pair)
            self.all_instances.append(pairs)
      
          #print("all_instances=",all_instances)
          #print("all_instances[0]=",all_instances[0])
      
          for i in self.all_instances[0]:
            print(i.visual_instance)
            
          #sync transforms
          #for pairs in self.all_instances:
          #  print(len(pairs))
          #  for pair in pairs:
          #    print("pair.link=", pair.link_index, " pair.visual_instance=", pair.visual_instance)
          
      
          print("len(self.all_instances)=",len(self.all_instances))
          print("\nhold CTRL and right/left/middle mouse button to rotate/zoom/move camera")
          
          st = time.time()
          
          #######################
          # create plane_visual_instance, a textured box plane

          # create a texture
          width = 256
          height = 256
          pixels = [255] * width * height * 3
          colorR = 255
          colorG = 255
          colorB = 255
          
          for i in range(width):
            for j in range(height):
               a = i < width / 2
               b = j < width / 2
               if (a == b):
                  pixels[(i + j * width) * 3 + 0] = 0
                  pixels[(i + j * width) * 3 + 1] = 255
                  pixels[(i + j * width) * 3 + 2] = 255
               else:
                pixels[(i + j * width) * 3 + 0] = colorR
                pixels[(i + j * width) * 3 + 1] = colorG
                pixels[(i + j * width) * 3 + 2] = colorB
          
          
          textureIndex = self.viz.opengl_app.renderer.register_texture(pixels, width, height, False)
          shape = self.viz.opengl_app.register_cube_shape(128, 128, 0.01, textureIndex, 4000)
          print("shape=",shape)
          pos = g.TinyVector3f(0.,0.,0.)
          orn = g.TinyQuaternionf(0.,0.,0.,1.)
          color = g.TinyVector3f(1.,1.,1.)
          scaling = g.TinyVector3f(1.,1.,1.)
          opacity = 1
          rebuild = True
          plane_visual_instance = self.viz.opengl_app.renderer.register_graphics_instance(shape, pos, orn, color, scaling, opacity, rebuild)
          
          
          #######################
          
          
          if 1:
            width = self.viz.opengl_app.renderer.get_screen_width()
            #print("screen_width=",width)
            height = self.viz.opengl_app.renderer.get_screen_height()
            #print("screen_height=",height)
            
            tile_width = int(width/self.max_x)
            tile_height = int(height/self.max_x)
            self.tiles=[]
            x=0
            y=0
            for t in range (self.num_envs):
                tile = g.TinyViewportTile()
                pairs = self.all_instances[t]
                
                # each tile only renders its own 'viz_instances'
                viz_instances = []
                
                # add the robot specific to this 'tile' to its 'viz_instances'
                #for pair in pairs:
                #  viz_instances.append(pair.visual_instance)
                
                #add the single ground plane to each tile
                viz_instances.append(plane_visual_instance)
                
                #print("viz_instances=",viz_instances)
                tile.visual_instances = viz_instances#[t, 512+t, 1024+t]
                #print("tile.visual_instances=",tile.visual_instances)
                cam = self.viz.opengl_app.renderer.get_active_camera()
                tile.projection_matrix = cam.get_camera_projection_matrix()
                tile.view_matrix = cam.get_camera_view_matrix()
                tile.viewport_dims=[x*tile_width,y*tile_height,tile_width, tile_height]
                self.tiles.append(tile)
                x+=1
                if x>=self.max_x:
                  x=0
                  y+=1
      
            # example how to initialize a camera
            cam = g.TinyCamera()
            cam.set_camera_up_axis(2)
            cam.set_camera_distance(1.5)
            cam.set_camera_pitch(-30)
            cam.set_camera_yaw(0)
            cam.set_camera_target_position(0.,0.,1.)
            self.viz.opengl_app.renderer.set_camera(cam)
        
        # End setup EGL renderer
        
        
  
    @nvtx.annotate("compute_observations", color="blue")
    def compute_observations(self):
        """ Computes observations
        """
        
        if self.device != 'cpu':
          self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        
        if 1:
        
          #################################### 
          #print("A1.compute_observations")
          #rbs = self._rigid_body_state.cpu().numpy()
          #visual_world_transforms = np.array(rbs).copy()
          #print("visual_world_transforms.shape=",visual_world_transforms.shape)
          #print("visual_world_transforms=",repr(visual_world_transforms))
  
          rbs = self._rigid_body_state.cpu().numpy()
          visual_world_transforms = np.array(rbs).copy()
          visual_world_transforms = visual_world_transforms[:,:,:7]
          visual_world_transforms = visual_world_transforms.reshape(self.num_envs,17*7)
          #print("visual_world_transforms .shape=", visual_world_transforms.shape)
          #print("visual_world_transforms[0]=", repr(visual_world_transforms[0]))
    
      
  
          if self.tiled_egl:
            import pytinyopengl3 as g
            # Copy transforms to CPU and convert to layout that renderer expects
            
          
            # if use_tiled = False, it will render all setups with spacing, similar to IsaacGym
            use_tiled = True
            if use_tiled:
              sim_spacing = 0
            else:
              sim_spacing = 2
            
            ct = time.time()
            # Synchronize the renderer world transforms for all environments
            #self.viz.sync_visual_transforms(self.all_instances, visual_world_transforms, 0, sim_spacing, apply_visual_offset=True)
  
            et = time.time()
            #print("sync_visual_transforms dt=",et-ct)
  
            
            # dump_next_frame_to_png is very slow, use only for debugging
            #name = "test_"+str(frame)+".png"
            #self.viz.opengl_app.dump_next_frame_to_png(filename=name, render_to_texture=False, width=-1, height=-1)#19200, height=10800)
                      
            width = self.viz.opengl_app.renderer.get_screen_width()
            height = self.viz.opengl_app.renderer.get_screen_height()
          
            tile_width = int(width/self.max_x)
            tile_height = int(height/self.max_x)
            
            if use_cuda_interop:
              self.viz.opengl_app.enable_render_to_texture(self.width, self.height)
  
            
            
            tile_index = 0
            x=0
            y=0
            
            cam = self.viz.opengl_app.renderer.get_active_camera()  
            
            ct = time.time()
            
            if use_tiled:
                with nvtx.annotate("tiled_cam_update", color="green"):
                  cam_local_pose = g.TinyPosef(g.TinyVector3f(0.28,0,0.03), g.TinyQuaternionf(0.,0.,0.,1.))
                  for tile_index in range (self.num_envs):
                    tile = self.tiles[tile_index]
                    
                    # update the camera view matrix (an OpenGL 4x4 matrix)
                    torso_world_pos = g.TinyVector3f(visual_world_transforms[tile_index][0],visual_world_transforms[tile_index][1],visual_world_transforms[tile_index][2])
                    torso_world_orn = g.TinyQuaternionf(visual_world_transforms[tile_index][3],visual_world_transforms[tile_index][4],visual_world_transforms[tile_index][5],visual_world_transforms[tile_index][6])
                    torso_pose = g.TinyPosef(torso_world_pos,torso_world_orn)
                    cam_world_pose = torso_pose * cam_local_pose
                    
                    mat = g.TinyMatrix3x3f(cam_world_pose.orientation)
                    
                    cam_pos = cam_world_pose.position
                    cam_distance=1.5
                    cam_forward = mat.get_column(0)
                    cam_up = mat.get_column(2)
                    #print("cam_up=",cam_up)
                    cam_target = cam_pos + cam_forward * cam_distance
                    #print("cam_target=",cam_target)
                    view_mat = g.compute_camera_view_matrix(cam_pos, cam_target, cam_up)
                    tile.view_matrix = view_mat#cam.get_camera_view_matrix()
                    
                    tile.viewport_dims=[x*tile_width,y*tile_height,tile_width, tile_height]
                    x+=1
                    if x>=self.max_x:
                      x=0
                      y+=1
                      
   
            et = time.time()
            #print("tile update dt=",et-ct)
   
            ct = time.time()
                      
            if use_tiled:
                with nvtx.annotate("render_tiled", color="yellow"):
                  self.viz.render_tiled(self.tiles, do_swap_buffer = False, render_segmentation_mask=(self.show_data==SEGMENTATION_DATA))
            else:
              self.viz.render(do_swap_buffer=False, render_segmentation_mask=(self.show_data==SEGMENTATION_DATA))  
  
            et = time.time()
            #print("render dt=",et-ct)
              
            #self.viz.render()
      
            
            
            #ReadPixelBuffer should not read back data to CPU, but using GL to CUDA PyTorch interop
            if use_cuda_interop:
              ct = time.time()
              self.viz.opengl_app.cuda_copy_texture_image(self.cuda_tex, self.cuda_mem, self.cuda_num_bytes)
              et = time.time()
              #print("cuda_copy_texture_image dt=",et-ct)
            else:
              ct = time.time()
              pixels = g.ReadPixelBuffer(self.viz.opengl_app)
              et = time.time()
              #print("ReadPixelBuffer dt=",et-ct)
              
            
            #print('pixels.rgba=', pixels.rgba)
      
            if self.matplotlib:
              
              if use_cuda_interop:
                ftensor = self.ttensor.type(torch.float32)
                np_img_arr = ftensor.cpu().numpy()
                np_img_arr = np.reshape(np_img_arr, (self.height, self.width, 4))
                print("np_img_arr.shape=",np_img_arr.shape)
                np_img_arr = np.flipud(np_img_arr)
                self.matplot_image.set_data(np_img_arr)
              else:
                if self.show_data == DEPTH_DATA:
                  np_depth_arr = np.flipud(np.reshape(pixels.depth, (height, width, 1)))
                  self.matplot_image.set_data(np_depth_arr)
                else:
                  np_img_arr = np.reshape(pixels.rgba, (height, width, 4))
                  if self.show_data==RGB_DATA:
                    np_img_arr = np_img_arr * (1. / 255.)
                  np_img_arr = np.flipud(np_img_arr)
                  self.matplot_image.set_data(np_img_arr)
              
              
              self.plt.pause(0.0001)
              
            self.viz.swap_buffer()
            
        
        if 1:
          width = self.viz.opengl_app.renderer.get_screen_width()
          #print("screen_width=",width)
          height = self.viz.opengl_app.renderer.get_screen_height()
          #print("screen_height=",height)
              
          tile_width = int(width/self.max_x)
          tile_height = int(height/self.max_x)
          ttensor = self.ttensor.type(torch.float32)*255.
          ttensor  = torch.reshape(ttensor, (self.height, self.width, 4))
          #ttensor = torch.flipud(ttensor)
          ttensor = ttensor.reshape(self.max_x, tile_width, self.max_x, tile_height, 4)
          ttensor = ttensor.swapaxes(1,2)
          ttensor = ttensor.reshape(self.max_x*self.max_x, tile_width*tile_height*4)
          ttensor = ttensor[:self.num_envs,]
         
          #self.ttensor  = torch.reshape(self.ttensor, (self.height, self.width, 4))
          #self.ttensor = self.ttensor[:self.num_envs,]
          #print("ttensor.shape=", ttensor.shape)
        
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    #,
                                    #ttensor
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
