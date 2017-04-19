acc = []
for i in range(200):
    with open('simple_%s.csv' % str(i), 'r') as f:
        accuracy = f.readline().split()[-1][:6]
        if float(accuracy) >= 0.50:
            acc.append((i, accuracy))
for i in range(len(acc)):
    print (str(acc[i][0]) + ': ' + str(acc[i][1]) + '\n')
