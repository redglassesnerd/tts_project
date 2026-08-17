def run_simulation(self, job_id: str, text: str, output_path: str, progress_callback, **kwargs):
    print("run_simulation called with job_id:", job_id)

class Handler:
    def __init__(self):
        self.is_simulator = True
    
    def synthesize(self, job_id, text, output_path, progress_callback, **kwargs):
        if self.is_simulator:
            return run_simulation(self, job_id, text, output_path, progress_callback, **kwargs)

h = Handler()
job = {
    "job_id": "123",
    "text": "hello",
    "speed": 1.0
}
job_params = {k: v for k, v in job.items() if k not in ["job_id", "output_path", "progress_callback"]}
print("job_params:", job_params)

# Call synthesize
h.synthesize(job_id="123", output_path="out.wav", progress_callback=lambda x, y: None, **job_params)
