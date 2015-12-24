def get_type(v):
    return type(v).__name__

def enumerate_events(variables, values, i = 0):
    if i >= len(variables):
           return [{}]
        
    events = []
    sub_events = enumerate_events(variables, values, i + 1)
    for v in values[variables[i]]:
        for e in sub_events:
            new_e = e.copy()
            new_e.update({variables[i] : v})
            events.append(new_e)
                 
    return events
