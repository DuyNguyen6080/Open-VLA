import argparse

def parse_args():
    home_parser = argparse.ArgumentParser(description="Robot SET Home position")
    
    # Adds the --camera argument. Defaults to 0 (default system webcam)
    home_parser.add_argument(
        "--inputfile", 
        type=str, 
         
        required=True,
        
        help="path to cvs file"
        
    )
    home_parser.add_argument(
        "--outputfile", 
        type=str, 
         
        required=True,
        
        help="path to out picture"
        
    )
    
    
    args = home_parser.parse_args()
    
        
    return args
