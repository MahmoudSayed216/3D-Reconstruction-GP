class DEBUGGER_SINGLETON:

    active = False
    debug_count = 0
    log_count = 0


color_code_map = {
    'red': "\033[31m",
    'green': "\033[32m",
}

reset = "\033[0m"

def _log_colored(string: str, source, counter, color):
    color_code = color_code_map[color]
    print(f"{color_code}[{source}:{counter}]{reset}{string}")



def DEBUG(string: str, obj=None):
    if DEBUGGER_SINGLETON.active:
        counter_str = str(DEBUGGER_SINGLETON.debug_count).rjust(3, '0')
        source = 'DEBUGGER'
        color = 'red'
        if obj is not None:
            string = ' '+string+': '+str(obj)
            _log_colored(string, source, counter_str, color)
        else:
            _log_colored(string, source, counter_str, color)
        print("\n")
        DEBUGGER_SINGLETON.debug_count+=1

def LOG(string: str, obj= None):
    counter_str = str(DEBUGGER_SINGLETON.log_count).rjust(3, '0')
    source = 'LOGGER'
    color = 'green'
    if obj is not None:
        string = ' '+string+': '+str(obj)
        _log_colored(string, source, counter_str, color)
    else:
        _log_colored(string, source, counter_str, color)
    print("\n")
    DEBUGGER_SINGLETON.log_count+=1

