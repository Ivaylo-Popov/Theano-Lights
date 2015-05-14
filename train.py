import numpy as np
import time
from operator import add

from toolbox import *
from models import *

if __name__ == "__main__":

#   Hyper-parameters
#--------------------------------------------------------------------------------------------------
    hp = Parameters()
    with hp:
        batch_size = 1000
        test_batch_size = 1000
        train_perm = False

        load_model = False
        save_model = True

        debug = False
        resample_z = False

        #Model = ffn.FFN
        #Model = ffn_bn.FFN_bn
        #Model = cnn.CNN
        
        #Model = vae1.Vae1
        #Model = cvae.Cvae 
        #Model = draw_at_lstm1.Draw_at_lstm1 
        #Model = draw_at_lstm2.Draw_at_lstm2 
        Model = draw_lstm1.Draw_lstm1 
        #Model = draw_sgru1.Draw_sgru1 

        init_scale = 1.05  
        learning_rate = 0.0016 
        lr_halflife = 50

        ''' sgd(0.001),  rmsprop(0.001),  adam(0.0005),  adamgc(0.0005),  esgd(0.01) '''
        optimizer = adamgc

        # ------------------
        walkforward = False
        walkstep_size = 5
        ws_iterations = 200
        n_stepdecay = 1.0
        ws_validstop = 0.02


#   Data
#--------------------------------------------------------------------------------------------------
    data_path = 'data/'

    data = mnist(path=data_path+'mnist/', nvalidation=0) 
    #data = mnistBinarized(path=data_path+'mnist/')  # only for UL models

    #data = mnist(path=data_path+'mnist/', distort=3, shuffle=True) 
    #data = freyfaces(path=data_path+'frey/')

    #data = downsample(data)
    
    visualize(-1, data['tr_X'][0:min(len(data['tr_X']), 900)], data['shape_x'])


#   Training
#--------------------------------------------------------------------------------------------------
    model = Model(data, hp)
    
    print ("M: %s  lr: %.5f  init: %.2f  batch: %d  ws: %d  iter: %d" % (model.id, learning_rate, init_scale, batch_size, walkforward*walkstep_size, ws_iterations)) 
    
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

    else:
        # Full training data learning
        n_iterations = 10000
        freq_save = 20
        freq_sample = 10
        it_lr = learning_rate

        for it in range(n_iterations):
            begin = time.time()
            model.permuteData(data)
                
            tr_outputs = model.train_epoch(it_lr)
            if len(data['va_X']) > 0:
                te_outputs = model.validation_epoch()
            else:
                te_outputs = model.test_epoch()
                
            if model.type == 'SL':
                # Supervised learning
                print("%d,%.4f,%.4f,%.4f,%.4f,%.2e,%.2f" % (it, 
                                                tr_outputs[model.outidx['cost']], te_outputs[model.outidx['cost']],
                                                tr_outputs[model.outidx['error_map_pyx']], te_outputs[model.outidx['error_map_pyx']], 
                                                tr_outputs[model.outidx['norm_grad']],
                                                time.time() - begin))
            elif model.type == 'UL':
                # Unsupervised learning
                print("%d,%.2f,%.2f,%.2f,%.2f,%.2e,%.2f" % (it, 
                                                       tr_outputs[model.outidx['cost_q']], te_outputs[model.outidx['cost_q']], 
                                                       tr_outputs[model.outidx['cost']], te_outputs[model.outidx['cost']], 
                                                       tr_outputs[model.outidx['norm_grad']],
                                                       time.time() - begin))
                # Generate samples
                if it % freq_sample == 0:
                    y_samples = model.decode(36 * (model.n_t + 1))
                    y_samples = np.transpose(y_samples, (1,0,2)).reshape((-1, y_samples.shape[2]))
                    visualize(it, y_samples, data['shape_x'])

            # Save model parameters
            if hp.save_model and it % freq_save == 0:
                model.save()

            it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
