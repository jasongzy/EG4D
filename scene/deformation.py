import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField
from arguments import ModelHiddenParams
from scene.utils import nerf_encoding
from utils.general_utils import concrete_random

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=None, args: ModelHiddenParams = None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = [] if skips is None else skips

        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        self.args = args
        if self.args.predict_flow:
            self.velocity_head = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        if self.args.predict_confidence:
            self.confidence_head = nn.Sequential(nn.ReLU(), nn.Linear(self.W + 27, self.W), nn.ReLU(), nn.Linear(self.W, 1), nn.Sigmoid())
        if self.args.predict_transient:
            self.transient_head = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        if self.args.predict_color_deform:
            # self.color_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
            # self.color_deform_w = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1), nn.ReLU())
            self.color_deform_w = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1), nn.CELU(alpha=1.0))
            self.color_deform_b = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))

    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]
        for _ in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        # output_dim = self.W
        return (
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1)),
        )

    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None, time_extra=None, campos=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb, time_extra, campos)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb, time_extra=None, campos=None):
        hidden = self.query_time(rays_pts_emb, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        # hidden = self.query_time(rays_pts_emb, torch.zeros_like(rays_pts_emb)[:, 0:1]).float()
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr
        if not self.args.do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden if (self.training or self.args.render_ref_time < 0) else self.query_time(rays_pts_emb, torch.full_like(rays_pts_emb, self.args.render_ref_time)[:, 0:1]).float())
            opacity = opacity_emb[:,:1] + do
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        if self.args.predict_flow:
            velocity_hidden = hidden
            if time_extra is not None:
                velocity_hidden_all = [self.query_time(rays_pts_emb, t).float() for t in time_extra]
                velocity_hidden_all.append(velocity_hidden)
                velocity_hidden = torch.mean(torch.stack(velocity_hidden_all, dim=0), dim=0)
            velocity = self.velocity_head(velocity_hidden)
        else:
            velocity = None

        if self.args.predict_confidence and campos is not None:
            viewdir = rays_pts_emb - campos[None]
            viewdir = viewdir / torch.linalg.norm(viewdir, axis=-1, keepdims=True)
            viewdir = nerf_encoding(viewdir)
            confidence = self.confidence_head(torch.cat([hidden, viewdir], dim=-1))
        else:
            confidence = None

        if self.args.predict_transient:
            transient = self.transient_head(hidden if (self.training or self.args.render_ref_time < 0) else self.query_time(rays_pts_emb, torch.full_like(rays_pts_emb, self.args.render_ref_time)[:, 0:1]).float())
            transient = concrete_random(transient, uniform=None if (self.training or self.args.render_ref_time < 0) else 0.5)
            # transient = torch.sigmoid(transient - 10.0)
        else:
            transient = None

        if self.args.predict_color_deform:
            # color_deform = self.color_deform(hidden if (self.training or self.args.render_ref_time < 0) else self.query_time(rays_pts_emb, torch.full_like(rays_pts_emb, REF_TIME)[:, 0:1]).float())
            color_deform_w = self.color_deform_w(hidden if (self.training or self.args.render_ref_time < 0) else self.query_time(rays_pts_emb, torch.full_like(rays_pts_emb, self.args.render_ref_time)[:, 0:1]).float())
            color_deform_b = self.color_deform_b(hidden if (self.training or self.args.render_ref_time < 0) else self.query_time(rays_pts_emb, torch.full_like(rays_pts_emb, self.args.render_ref_time)[:, 0:1]).float())
            color_deform = torch.cat((color_deform_w, color_deform_b), axis=-1)
        else:
            color_deform = None

        return pts, scales, rotations, opacity, velocity, confidence, transient, color_deform

    def get_mlp_parameters(self):
        parameter_list = [param for name, param in self.named_parameters() if "grid" not in name]
        return parameter_list

    def get_grid_parameters(self):
        return list(self.grid.parameters() ) 
    # + list(self.timegrid.parameters())


class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # self.deformation_net.opacity_deform.apply(initialize_weights_zeros)
        if args.predict_transient:
            self.deformation_net.transient_head.apply(initialize_weights_zeros)
        # if args.predict_color_deform:
        #     # self.deformation_net.color_deform.apply(initialize_weights_zeros)
        #     self.deformation_net.color_deform_w.apply(initialize_weights_zeros)
        #     self.deformation_net.color_deform_b.apply(initialize_weights_zeros)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, time_extra=None, campos=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel, time_extra, campos)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None, time_extra=None, campos=None):
        return self.deformation_net(
            point,
            scales,
            rotations,
            opacity,
            times_sel,
            time_extra,
            campos,
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def initialize_weights_zeros(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            init.constant_(m.bias, 0)