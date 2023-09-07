import csv

# Get DMC Threads
class Thread:
    def __init__(self, DMCcode, coloursName, hexValue):
        self.code = DMCcode
        self.name = coloursName
        h = hexValue.lstrip('#')
        self.rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    def __str__(self):
        return f"DMC: {self.code}\t Name: {self.name}\t\t\t\t RGB: {self.rgb}"

def load_threads():
    threads = []
    with open('DMC_threads.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            threads.append(Thread(row[0], row[1], row[2][1:]))
    return threads