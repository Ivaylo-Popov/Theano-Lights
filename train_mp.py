import numpy as np
import time
from operator import add
import random
import math

from toolbox import *
from models import *


if __name__ == "__main__":

#   Hyper-parameters
#--------------------------------------------------------------------------------------------------
    hp = Parameters()
    with hp:
        batch_size = 64
        test_batch_size = 64
        
        load_model = False
        save_model = True

        debug = False
        resample_z = False
        train_perm = False
        walkforward = False
        dynamic_eval = False

        #Model = mp_ffn.MP_ffn
        #Model = mp_lstm.MP_lstm
        Model = mp_rnn.MP_rnn

        seq_size = 20
        warmup_size = 1

        init_scale = 1.1
        learning_rate = 0.1
        lr_halflife = 500
        optimizer = sgdgc

        description = ''


#   Data
#--------------------------------------------------------------------------------------------------
    data_path = 'data/'

    data = us_futures(path=data_path+'finance/',  trading_inputs_name='D:/Work/Quant-Lights/instruments/trading_inputs3.txt',
                      name='us_futures_r_norm3_3np', batch_size=batch_size, n_train=-1, n_valid=5*64)

    visualize_tokens(-1, 
                     (data['tr_X'][0, :, 6]-np.min(data['tr_X'][0, :, 6], axis=0, keepdims=True))/
                     (np.max(data['tr_X'][0, :, 6], axis=0, keepdims=True)-np.min(data['tr_X'][0, :, 6], axis=0, keepdims=True)), 
                     data['shape_x'])
    
#   Training
#--------------------------------------------------------------------------------------------------
    model = Model(data, hp)
    
    print ("M: %s  lr: %.5f  init: %.2f  batch: %d  seq_size: %d  desc: %s" % (model.id, learning_rate, init_scale, batch_size, seq_size, description)) 
    
    if walkforward:
        # Walkforward learning
        n_ws = len(data['tr_X']) / walkstep_size / batch_size
        it_lr = learning_rate
        
        for walkstep in xrange(0, n_ws):
            begin = time.time()
            min_validation = 100000.
            
            #tr_outputs = model.train_walkstep(walkstep, ws_iterations, it_lr)

            # Validate on previous data
            for it in range(ws_iterations):
                begin_inner = time.time()
                tr_outputs = None
                
                for i in xrange(0, walkstep_size):
                    batch_idx = walkstep * walkstep_size + i
                    outputs = model.train(batch_idx, it_lr)
                    outputs = map(lambda x: x / float(walkstep_size * batch_size), outputs)
                    if i==0:
                        tr_outputs = outputs
                    else:
                        tr_outputs = map(add, tr_outputs, outputs)

                prev_va_outputs = [0.] * 100
                for i in xrange(0, walkstep * walkstep_size):
                    outputs = model.validate(i)
                    outputs = map(lambda x: x / (walkstep * walkstep_size * batch_size), outputs)
                    prev_va_outputs = (map(add, prev_va_outputs, outputs) if i!=0 else outputs)
                
                print(" > %d,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.3f,\t%.2f" % (it, 
                                                               tr_outputs[model.outidx['cost_q']], prev_va_outputs[model.outidx['cost_q']], 
                                                               tr_outputs[model.outidx['cost']], prev_va_outputs[model.outidx['cost']], 
                                                               tr_outputs[model.outidx['norm_grad']], 
                                                               time.time() - begin_inner))

                # Early stopping on previous data
                if prev_va_outputs[model.outidx['cost']] < min_validation:
                    min_validation = prev_va_outputs[model.outidx['cost']]
                elif prev_va_outputs[model.outidx['cost']] > min_validation * (1. + ws_validstop):
                    break


            te_outputs = model.test_epoch()

            if model.type == 'SL':
                # Supervised learning
                print("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f" % (walkstep, 
                                                tr_outputs[model.outidx['cost']], te_outputs[model.outidx['cost']],
                                                tr_outputs[model.outidx['error_map_pyx']], te_outputs[model.outidx['error_map_pyx']], 
                                                tr_outputs[model.outidx['norm_grad']],
                                                time.time() - begin))
            else:
                # Unsupervised learning
                print("%d,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.3f,\t%.2f" % (walkstep, 
                                                       tr_outputs[model.outidx['cost_q']], te_outputs[model.outidx['cost_q']], 
                                                       tr_outputs[model.outidx['cost']], te_outputs[model.outidx['cost']], 
                                                       tr_outputs[model.outidx['norm_grad']],
                                                       time.time() - begin))
                
                # Generate samples
                y_samples = model.decode(36 * (model.n_t + 1))
                y_samples = np.transpose(y_samples, (1,0,2)).reshape((-1, y_samples.shape[2]))
                visualize(walkstep + 1, y_samples, data['shape_x'])

            #it_lr = float(learning_rate / (walkstep + 1.))
            #it_lr = it_lr*n_stepdecay
            ws_iterations = int(ws_iterations*n_stepdecay)

            model.save()

    elif dynamic_eval:
        # Dynamic evaluation
        it_lr = float(learning_rate)

        va_outputs = model.validation_epoch()
        te_outputs = model.test_epoch()

        dyn_va_outputs = model.dyn_validation_epoch(it_lr)
        model.load()
        dyn_te_outputs = model.dyn_test_epoch(it_lr)
                
        if model.type == 'LM':
            # Lanugage model
            print("0,%.2f,%.2f,%.2f,%.2f" % (np.exp(va_outputs[model.outidx['cost']]), 
                                             np.exp(te_outputs[model.outidx['cost']]),
                                             np.exp(dyn_va_outputs[model.outidx['cost']]), 
                                             np.exp(dyn_te_outputs[model.outidx['cost']]) ))
            
    else:
        # Full training data learning
        n_iterations = 10000
        freq_save = 2
        freq_sample = 2
        it_lr = float(learning_rate)
        #rnd_offset = np.arange(seq_size)
        rnd_offset = np.random.permutation(seq_size)
        #rnd_offset = [0] * seq_size

        def r2sqrt(y, y_hat):
            res = np.sqrt((y-y_hat)/y)
            res = np.nan_to_num(res)
            return res

        for it in range(n_iterations):
            begin = time.time()
            model.permuteData(data)
                
            tr_outputs = model.train_epoch(it_lr, rnd_offset[it % seq_size])
            va_outputs = model.validation_epoch()
            #te_outputs = model.test_epoch()
            
            np.savetxt('temp.csv', np.column_stack((
                r2sqrt(tr_outputs[2][:,1],tr_outputs[2][:,0]), 
                r2sqrt(va_outputs[2][:,1],va_outputs[2][:,0]))), delimiter=',')

            tr_sharpe = r2sqrt(tr_outputs[model.outidx['cost_base']], tr_outputs[model.outidx['cost']])
            va_sharpe = r2sqrt(va_outputs[model.outidx['cost_base']], va_outputs[model.outidx['cost']])

            if model.type == 'MP':
                # Lanugage model
                print("%d,%.4e,%.4e,%.4e,%.4e,%.4f,%.4f,%.2e,%.2f" % (it, 
                                                tr_outputs[model.outidx['cost']], 
                                                va_outputs[model.outidx['cost']],
                                                tr_outputs[model.outidx['cost_base']], 
                                                va_outputs[model.outidx['cost_base']],
                                                tr_sharpe,
                                                va_sharpe,
                                                tr_outputs[model.outidx['norm_grad']],
                                                time.time() - begin))
                # Generate samples
                #samples = model.decode(random.randint(0, data['n_tokens']-1))

            # Save model parameters
            if hp.save_model and it % freq_save == 0:
                model.save()
            
            if lr_halflife != 0:
                it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
                