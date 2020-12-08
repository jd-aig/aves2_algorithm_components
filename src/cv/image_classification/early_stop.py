def loss_early_stop(loss, param):
    #print('Last Loss {}; Loss Count:{}'.format(param['last'],param['count']))
    if loss >= param['last']:
        param['count'] += 1
    else:
        param['count'] = 0
    param['last'] = loss
    return param
  
def top1_early_stop(top1, param):
  if top1 > param['top1_max']:
    param['top1_max'] = top1
    param['count'] = 0 
  else:
    param['count'] += 1
  return param