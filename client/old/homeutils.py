import argparse

def parse_args():
    home_parser = argparse.ArgumentParser(description="Robot SET Home position")
    
    # Adds the --camera argument. Defaults to 0 (default system webcam)
    home_parser.add_argument(
        "--posmode", 
        type=str, 
         
        required=True,
        choices=["j", "tcp"],
        help="home position mode --posmode j for joint and tcp for tcp position "
        
    )
    
    
    args = home_parser.parse_args()
    
        
    return args
