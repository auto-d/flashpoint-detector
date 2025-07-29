import subprocess

def run_subprocess(cmd, output=False, output_file=None): 
    """
    Shell out to run a provided command, optionally returning stdout TEXT or redirecting stdout
    TEXT to a provided file.

    NOTE: repurposed from NLP submission
    """
    kwargs = {}
    file = None     
    if output_file:                 
        file = open(output_file, "w+")
        kwargs['stdout'] = file
    else: 
        kwargs['capture_output']=output

    try:
        result = subprocess.run(cmd,text=True, **kwargs)
        return True, result.stdout
    
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    
    finally: 
        if file: 
            file.close()
        
    
        
    