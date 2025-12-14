def pretty_header(func):
    def enhanced(*args,**kwargs):
        print("\n\n"+"-----*"*3+" "+func.__name__+"  "+"*-----"*3)
        result = func(*args,**kwargs)
        # print("-----*"*6+"-----")
        return result
    return enhanced

def pretty_print(func):
    def enhanced(*args,**kwargs):
        print("\n\n"+"-----*"*3+" "+func.__name__+"  "+"*-----"*3)
        result = func(*args,**kwargs)
        print("-----*"*6+"-----")
        return result
    return enhanced

def pretty_footer(func):
    def enhanced(*args,**kwargs):
        # print("\n\n"+"-----*"*3+" "+func.__name__+"  "+"*-----"*3)
        result = func(*args,**kwargs)
        print("-----*"*6+"-----")
        return result
    return enhanced
