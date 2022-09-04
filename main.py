from fastapi import FastAPI
from dataclasses import dataclass
from typing import List
from fastapi import Response


from parcels import FieldSet, Field, ParticleSet, Variable, JITParticle
import numpy as np
from parcels import AdvectionRK4, plotTrajectoriesFile, ErrorCode

import math
import json
import xarray as xr
from datetime import timedelta

app = FastAPI()
field_set = None
data_name = 'hk_cleaned.nc'
temp_file_name = 'temporary.nc'

@dataclass
class Particles:
    # [[lon, lat], ...]
    particles: List[List[float]]

def DeleteParticle(particle, fieldset, time):
    particle.delete()

@app.on_event("startup")
async def startup_event():
    global field_set
    # Set up fieldset
    variables = {'U': 'u',
                 'V': 'v',
                 'depth': 'w'
                 }

    dimensions = {'lat': 'lat_rho',
                  'lon': 'lon_rho',
                  }

    field_set = FieldSet.from_c_grid_dataset(data_name, variables, dimensions, allow_time_extrapolation=True)

@app.get("/")
async def root():
    return {"message": "Welcome"}

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

@app.post("/execute")
async def execute(input_particles: Particles):

    #Set up particle set
    num_particles = len(input_particles.particles)
    if num_particles == 0:
        return None
    particles_list = np.array(input_particles.particles)
    lon = particles_list[:, 0]
    lat = particles_list[:, 1]
    pset = ParticleSet(fieldset=field_set, pclass=JITParticle, lat= lat, lon=lon)

    #Advect
    output_file = pset.ParticleFile(name=temp_file_name,
                                    outputdt=timedelta(hours=2))
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=30),
                 dt=timedelta(minutes=5),
                 output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                 )

    output_file.close()

    #Read
    parcels = xr.open_dataset(temp_file_name)
    # print(list(parcels["lon"].to_numpy()))
    # print(type(list(parcels["lon"].to_numpy())))
    lons = parcels["lon"].to_numpy()
    lats = parcels["lat"].to_numpy()

    # print(lons, lats)

    trajectories = []
    for i in range(num_particles):
        trajectories.append(list(zip(lons[i], lats[i])))


    # for (lon, lat) in zip(lons, lats):
    #     trajectories

    print(trajectories)

    json_data = json.dumps({'trajectories': trajectories})
    return Response(content=json_data, media_type="application/json")
