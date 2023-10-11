import json

import numpy as np
from qiskit import IBMQ

account = ''

IBMQ.save_account(account, hub='', group='', project='', overwrite=True)
provider = IBMQ.load_account()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.iscomplexobj(obj):
            return abs(obj)
        return json.JSONEncoder.default(self, obj)


file_with_jobs = 'jobs_details/jobs_to_store.txt'

with open(file_with_jobs, 'r') as f:
    for job_id in f.readlines():
        job_id = job_id.strip(' \n')
        job = provider.runtime.job(job_id)
        result = job.result()

        with open(f'jobs_details/{job_id}.json', 'w', encoding='utf-8') as f2:
            json.dump(result, f2, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            print(job_id)
