avoid = {'vaish': ['aayush', 'kaela'], 'aayush': [], 'nathan': ['kaela', 'vaish']}
assign = {'vaish': None, 'aayush': None, 'kaela': None, 'Nathan': None}
available = ['vaish', 'aayush', 'nathan', 'kaela']

def match(remaining, person, avoid):
    for p in avoid[person]:
        for p in people:
            assign[people]