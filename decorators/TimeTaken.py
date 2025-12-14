import time

def time_taken(base_fun):
    """
        Decorator to get execution time of any function
    """
    def modified_fun(*args,**kwargs):
        start = time.time()
        
        result = base_fun(*args,**kwargs)
        
        end = time.time()

        print(f"Time taken:{end-start}")
        return result

    return modified_fun
