from metaflow import FlowSpec, step, IncludeFile, Parameter, conda_base,schedule
import os


global_value = 42

@schedule(daily=True)
class ProcessDemoFlow(FlowSpec):
    
    @step
    def start(self):
        global global_value
        global_value = 43
        print('global_value:', global_value)
        print('process ID:', os.getpid())
        self.next(self.end)
    
    @step
    def end(self):
        print('global_value:', global_value)
        print('process ID:', os.getpid())
        
if __name__ == '__main__':
    ProcessDemoFlow()