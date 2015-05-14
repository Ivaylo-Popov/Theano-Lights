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
        dynamic_eval = False

        #Model = mp_ffn.MP_ffn
        #Model = mp_lstm.MP_lstm
        Model = me_rnn.ME_rnn
        #Model = me_rnn2.ME_rnn2
        #Model = me_rnn_bn.ME_rnn_bn

        seq_size = 50
        warmup_size = 1

        init_scale = 1.1
        learning_rate = 0.05
        lr_halflife = 500
        optimizer = sgdgc

        description = ''

        # ------------------
        walkforward = False
        walkstep_size = 1
        ws_iterations = 20
        n_stepdecay = 1.0
        ws_validstop = 0.02


#   Data
#--------------------------------------------------------------------------------------------------
    data_path = 'data/'

    #data = us_futures(path=data_path+'finance/', name='us_futures4_r2_21', 
    #                  trading_inputs_name='D:/Work/Quant-Lights/instruments/trading_inputs2_1.txt', batch_size=batch_size, n_train=-1, n_valid=5*64)
   
    #data = us_futures(path=data_path+'finance/', name='us_futures4_r2_3',
    #                  trading_inputs_name='D:/Work/Quant-Lights/instruments/trading_inputs3.txt', batch_size=batch_size, n_train=-1, n_valid=5*64)

    #data = us_futures(path=data_path+'finance/', name='us_futures4a',
    #                  trading_inputs_name='D:/Work/Quant-Lights/instruments/trading_inputs3.txt', batch_size=batch_size, n_train=-1, n_valid=3*64)  # 18*64, n_valid=1*32)

    data = us_futures(path=data_path+'finance/', name='eu_futures1a_xeu',
                      trading_inputs_name='D:/Work/Quant-Lights/instruments/eu_trading_inputs1.txt', batch_size=batch_size, n_train=-1, n_valid=3*64)  # 18*64, n_valid=1*32)

    visualize_tokens(-1, 
                     (data['tr_X'][0, :, 0]-np.min(data['tr_X'][0, :, 0], axis=0, keepdims=True))/
                     (np.max(data['tr_X'][0, :, 0], axis=0, keepdims=True)-np.min(data['tr_X'][0, :, 0], axis=0, keepdims=True)), 
                     data['shape_x'])
    
#   Training
#--------------------------------------------------------------------------------------------------
    model = Model(data, hp)
    
    print ("M: %s  lr: %.5f  init: %.2f  batch: %d  seq_size: %d  desc: %s" % (model.id, learning_rate, init_scale, batch_size, seq_size, description)) 
    
    if walkforward:
        # Walkforward learning
        n_ws = len(data['tr_X']) / walkstep_size
        it_lr = learning_rate
        rnd_offset = np.arange(seq_size)

        for walkstep in xrange(0, n_ws):
            begin = time.time()
            min_validation = 100000.
            
            # Validate on previous data
            for it in range(ws_iterations):
                tr_outputs = None
                
                offset = rnd_offset[it % seq_size]  
                if it > 0:
                    model.reset_hiddenstates()

                num_batches = walkstep_size * ((data['tr_X'].shape[1] - offset) / seq_size)
                seq_per_epoch = batch_size * (seq_size - warmup_size) * num_batches

                for i in xrange(0, num_batches):
                    group_idx = walkstep * walkstep_size + i / ((data['tr_X'].shape[1] - offset) / seq_size)
                    seq_idx = i % ((data['tr_X'].shape[1] - offset) / seq_size)
                    outputs = model.train(group_idx, seq_idx, it_lr, offset)
                    outputs = map(lambda x: x / float(seq_per_epoch), outputs)
                    if i==0:
                        tr_outputs = outputs
                    else:
                        tr_outputs = map(add, tr_outputs, outputs)
               
                if it==0:
                    tr_outputs0 = tr_outputs

                ##
                #prev_va_outputs = [0.] * 100
                #for i in xrange(0, walkstep * walkstep_size):
                #    outputs = model.validate(i)
                #    outputs = map(lambda x: x / (walkstep * walkstep_size * batch_size), outputs)
                #    prev_va_outputs = (map(add, prev_va_outputs, outputs) if i!=0 else outputs)
                
                #tr_sharpe = -tr_outputs[model.outidx['cost']] / np.sqrt(tr_outputs[model.outidx['cost_base']])

                #print(" > %d,%.4f,%.4f,%.4f,%.4f" % (it, 
                #                                     -tr_outputs[model.outidx['cost']], 0, 
                #                                     tr_sharpe, 0))

                # Early stopping on previous data
                #if prev_va_outputs[model.outidx['cost']] < min_validation:
                #    min_validation = prev_va_outputs[model.outidx['cost']]
                #elif prev_va_outputs[model.outidx['cost']] > min_validation * (1. + ws_validstop):
                #    break
                ##


            va_outputs = model.validation_epoch()

            tr_sharpe0 = -tr_outputs0[model.outidx['cost']] / np.sqrt(tr_outputs0[model.outidx['cost_base']])
            tr_sharpe = -tr_outputs[model.outidx['cost']] / np.sqrt(tr_outputs[model.outidx['cost_base']])
            va_sharpe = -va_outputs[model.outidx['cost']] / np.sqrt(va_outputs[model.outidx['cost_base']])

            if model.type == 'MP':
                # Market prices
                print("%d,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2e,%.2f" % (walkstep, 
                                                -tr_outputs[model.outidx['cost']], 
                                                -tr_outputs0[model.outidx['cost']], 
                                                -va_outputs[model.outidx['cost']],
                                                tr_sharpe, tr_sharpe0, va_sharpe,
                                                tr_outputs[model.outidx['norm_grad']],
                                                time.time() - begin))
                
            #it_lr = float(learning_rate / (walkstep + 1.))
            it_lr = it_lr*n_stepdecay
            #ws_iterations = int(ws_iterations*n_stepdecay)

            model.save()

    elif dynamic_eval:
        # Dynamic evaluation
        it_lr = float(learning_rate)

        va_outputs = model.validation_epoch()
        #te_outputs = model.test_epoch()

        dyn_va_outputs = model.dyn_validation_epoch(it_lr)
        #model.load()
        #dyn_te_outputs = model.dyn_test_epoch(it_lr)
                
        #np.savetxt('temp.csv', np.column_stack((tr_outputs[2], va_outputs[2])), delimiter=',')

        va_sharpe = -va_outputs[model.outidx['cost']] / np.sqrt(va_outputs[model.outidx['cost_base']])
        dyn_va_sharpe = -dyn_va_outputs[model.outidx['cost']] / np.sqrt(dyn_va_outputs[model.outidx['cost_base']])

        if model.type == 'MP':
            # Lanugage model
            print("0,%.4e,%.4e,%.4f,%.4f" % (-va_outputs[model.outidx['cost']], 
                                             -dyn_va_outputs[model.outidx['cost']], 
                                             va_sharpe, 
                                             dyn_va_sharpe))
            
    else:
        # Full training data learning
        n_iterations = 10000
        freq_save = 2
        freq_sample = 2
        it_lr = float(learning_rate)
        rnd_offset = np.arange(seq_size)
        #rnd_offset = np.random.permutation(seq_size)
        #rnd_offset = [0] * seq_size

        for it in range(n_iterations):
            begin = time.time()
            model.permuteData(data)
                
            tr_outputs = model.train_epoch(it_lr, rnd_offset[it % seq_size])
            va_outputs = model.validation_epoch()
            #te_outputs = model.test_epoch()
            
            np.savetxt('temp.csv', np.column_stack((tr_outputs[2], va_outputs[2])), delimiter=',')

            tr_sharpe = -tr_outputs[model.outidx['cost']] / np.sqrt(tr_outputs[model.outidx['cost_base']])
            va_sharpe = -va_outputs[model.outidx['cost']] / np.sqrt(va_outputs[model.outidx['cost_base']])

            if model.type == 'MP':
                # Market prices
                print("%d,%.2f,%.2f,%.4f,%.4f,%.2e,%.2f" % (it, 
                                                -tr_outputs[model.outidx['cost']],
                                                -va_outputs[model.outidx['cost']],
                                                tr_sharpe,
                                                va_sharpe,
                                                tr_outputs[model.outidx['norm_grad']],
                                                time.time() - begin))

            # Save model parameters
            if hp.save_model and it % freq_save == 0:
                model.save()

            exposure = model.va_exposure_seq()
            np.savetxt('exposure.csv', exposure, delimiter=',')
            #np.savetxt('exposure.csv', exposure.reshape((-1, 500, 2)).transpose(2, 0, 1).reshape((-1, 500)), delimiter=',')
            
            if lr_halflife != 0:
                it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
