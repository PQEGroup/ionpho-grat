import numpy as np
import tidy3d as td
import matplotlib.pyplot as plt
import mat73
# Define the file name

#fname = "data/oj80_375_20deg_fdve-a38a754c-3092-4e33-9bea-a3f562f51727.hdf5"
#mat = sio.loadmat('perDC_interp_initial_375_20degs.mat')

#foldername = "grats/lam397nm_thetm45_w0x1p5_w0y1p5_z50um/"
#output_sim_file = "lam397nm_thetm45_w0x1p5_w0y1p5_z50um"
foldername = "AIM/L397/jay_test/"
output_sim_file = "perDC_interp_initial_v7p3_lam397nm_thetm43_w0x0p51_w0y0p5_z030um_zOffXY0_0_tapStartm56p539822723741665_zX00umzY50um_useAlphasAir_v73"
addl_str = ""
mat = mat73.loadmat(foldername+'perDC_interp_initial_v7p3_lam397nm_thetm43_w0x0p51_w0y0p5_z030um_zOffXY0_0_tapStartm56p539822723741665_zX00umzY50um_useAlphasAir_v73.mat')

outl_str = "z="

p = mat["p"]
# Load the simulation data

sim_data = td.SimulationData.from_file(foldername+output_sim_file+addl_str+".hdf5")

Ex = (sim_data["field"].Ex)  # field is the name of the monitor
x = np.asarray(Ex.x)
y = np.asarray(Ex.y)
Ex = np.asarray(sim_data["field"].Ex)
Ey = np.asarray(sim_data["field"].Ey)
Ez = np.asarray(sim_data["field"].Ez)
Hx = np.asarray(sim_data["field"].Hx)
Hy = np.asarray(sim_data["field"].Hy)
Hz = np.asarray(sim_data["field"].Hz)

#print(np.asarray(Ex))
I = np.abs(np.multiply(Ex, np.conj(Ex)) + np.multiply(Ey, np.conj(Ey)) + np.multiply(Ez, np.conj(Ez)))
print(np.squeeze(I).shape, x.shape, y.shape)

# Define the farfield parameters
# radial distance away from the origin at which to project fields
r_proj = 28.5 * 1 #0.422
outl_str += str(r_proj)

# theta and phi angles at which to observe fields - part of the half-space to the right
theta_proj = np.linspace(-np.pi / 2+np.pi/20, np.pi/2-np.pi/20, 100)
phi_proj = np.linspace(-np.pi, np.pi , 100)

print(sim_data.simulation.monitors[0])
monitor_near = sim_data.simulation.monitors[0]
#monitor_near = sim_data["field"]
lambda0 = p['lambda']
freq0 = td.C_0 / lambda0

# How to calculate these and do it in parallel?
x_proj = np.linspace(30, 45, 100)
y_proj = np.linspace(-5.0, 5.0, 25)
y_proj = np.linspace(-0.02, 0.02, 2)




toxt = p['tOx']
if p['material'] == 'AO_AIM':
    boxt = 3.06+0.15+0.13 #um
elif p['material'] == 'FN_AIM':
    boxt = 3.06
tAir = 1.5 #um

sim_center = [9.105744940391286, 0.0, 2.21]
#sim_size = [19.211489880782572, 10.440627181318419, 12.52]

wg_excess = 1
tapl_SI = mat["tapl"]
gratl = mat["lgrat"]
wg_end = gratl + tapl_SI + wg_excess
sim_center = [0,0,0]
sim_center[0] = (wg_end+1)/2 # the +1 is from looks like what i've been passing to create_taper function
sim_center_0 = sim_center[0]

sim_size_0 = (wg_end+1)
print(sim_size_0)
# we don't need sim_center[2] here
#sim_center[2] = (wgh_low*1e6 + toxt-boxt+tAir-1.0) / 2
FF_approx = False
if FF_approx:
    outl_str += "_FFapprox"
else:
    outl_str += "_FFexact"
# far field projection monitor Cartesian
monitor_far = td.FieldProjectionCartesianMonitor(
    center=[sim_center[0], 0, toxt+tAir/2],
    # and angles will all be measured with respect to this local origin
    size=[td.inf, td.inf,0],
    # the size and center of any far field monitor should indicate where the *near* fields are recorded
    freqs=[freq0],
    name="far_field",
    x=list(x_proj),
    y=list(y_proj),
    proj_distance=r_proj,
    proj_axis = 2,
    far_field_approx=FF_approx,  # we leave this to its default value of 'True' because we are interested in fields sufficiently
    # far away that geometric far field approximations can be invoked to speed up the calculation
)

#monitor_near = td.FieldMonitor(center=[sim_center[0], 0, toxt+tAir/2], size=[40,40, 0], freqs=[freq0], name="field")


# helper functin to call the projector
def get_proj_fields2(sim_data, monitor_near, monitor_far, pts_per_wavelength=10):
    # object that does projections is constructed using the near-field monitor, because those are the fields to be projected
    projector = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[monitor_near],
        normal_dirs=["+"],  # we are projecting along the + direction
        pts_per_wavelength=pts_per_wavelength,  # to speed up calculations, the fields on the near-field monitor can be downsampled to these
        # many points per wavelength (default is already 10)
    )
    return projector.project_fields(monitor_far)


# execute the projector, with the far field monitor as input, to do the projection
# let's also time this, for later use
import time

t0 = time.perf_counter()
projected_field_data = get_proj_fields2(sim_data, monitor_near, monitor_far)
t1 = time.perf_counter()
proj_time = t1 - t0

def make_field_plot(phi, theta, vals1):
    n_plots = 1
    fig, ax = plt.subplots(1, n_plots, tight_layout=True, figsize=(8, 3.8))
    im1 = ax.pcolormesh(
        phi,
        theta,
        np.real(vals1),
        cmap="RdBu",
        shading="auto",
    )

    fig.colorbar(im1, ax=ax)
    ax.set_title("Analytic")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    #ax.set_aspect('equal')




# plot Etheta
#print(dir(projected_field_data))
print(projected_field_data.fields_cartesian)
print(projected_field_data.fields_cartesian.Ex)
print(projected_field_data.fields_cartesian.Ex.x)

Etheta_proj = projected_field_data.Etheta.isel(f=0)
Ephi_proj = projected_field_data.Ephi.isel(f=0)
Er_proj = projected_field_data.Er.isel(f=0)
I_proj = np.multiply(Etheta_proj, np.conj(Etheta_proj)) + np.multiply(Ephi_proj, np.conj(Ephi_proj)) + np.multiply(Er_proj, np.conj(Er_proj))
#print(I_proj.shape, Ephi_proj.shape, x_proj.shape, y_proj.shape)
#make_field_plot(y_proj, x_proj+0*sim_center[0], np.squeeze(abs(I_proj)))
make_field_plot(y_proj, x_proj+sim_center[0], np.squeeze(abs(I_proj)))


np.savez(foldername+outl_str, I_proj=I_proj, x_proj=x_proj, y_proj=y_proj, Etheta_proj=Etheta_proj, Ephi_proj=Ephi_proj, Er_proj=Er_proj, sim_center=sim_center, sim_size=sim_size_0, proj_time=proj_time, r_proj=r_proj, FF_approx=FF_approx)

plt.show()
