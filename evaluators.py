from collections import Counter

def getAccuracy(determined_Y, real_Y):
  real_dict = dict(Counter(real_Y))
  determined_dict = dict(Counter(determined_Y))

  # print("Actual number of persons under category that make over 50K : %s \n" % (real_dict.get(' >50K')))
  # print("Number of persons Classified under category that make over 50K : %s \n" % (determined_dict.get(' >50K')))
  # print("Actual number of persons under category that make less/equal to 50K : %s \n" % (real_dict.get(' <=50K')))
  # print("Number of persons Classified under category that make less/equal to 50K : %s \n" % (determined_dict.get(' <=50K')))

  print(" Category \t Actual \t Determined \n")
  print(" >50K     \t %s     \t %s \n" %(real_dict.get(' >50K'),determined_dict.get(' >50K')))
  print(" <=50K    \t %s     \t %s \n" %(real_dict.get(' <=50K'),determined_dict.get(' <=50K')))
  correct = 0 
  for x in range(len(real_Y)):
    if real_Y[x] == determined_Y[x]:
      correct += 1
  return 100.0*correct/float(len(real_Y))

def true_positives(determined_Y, real_Y, label):
  true_positives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] == label and real_Y[ii] == label: 
      true_positives+=1
  return true_positives

def all_positives(determined_Y, label):
  return Counter(determined_Y)[label]

def false_negatives(determined_Y, real_Y, label):
  false_negatives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] != label and real_Y[ii] == label: 
      false_negatives+=1
  return false_negatives
  
def precision(determined_Y, real_Y, label):
    if float(all_positives(determined_Y, label)) == 0: return 0
    return true_positives(determined_Y, real_Y, label) / float(all_positives(determined_Y, label))

def recall(determined_Y, real_Y, label):
    denominator = float((true_positives(determined_Y, real_Y, label) + false_negatives(determined_Y, real_Y, label)))
    if denominator == 0: return 0
    return true_positives(determined_Y, real_Y, label) / denominator

def f1_score(determined_Y, real_Y, label = 1):
    p = precision(determined_Y, real_Y, label)
    r = recall(determined_Y, real_Y, label)
    if p + r == 0: return 0
    f1 = 2 * (p * r) / (p + r)
    return f1
