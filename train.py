import numpy as np
import time

from toolbox import *
from models import *

if __name__ == "__main__":

#   Hyper-parameters
#--------------------------------------------------------------------------------------------------
    hp = Parameters()
    with hp:
        debug = False
        batch_size = 1000
        test_batch_size = 2000
        train_perm = False
        resample_z = False

        load_model = False
        save_model = True

        Model = ffn.FFN
        #Model = cnn.CNN
        
        #Model = vae1.Vae1
        #Model = cvae.Cvae 
        #Model = draw_at_lstm2.Draw_at_lstm2 
        #Model = draw_lstm1.Draw_lstm1 
        #Model = draw_sgru1.Draw_sgru1 

        init_scale = 1.0
        learning_rate = 0.0005

        ''' sgd(1),  rmsprop(0.001),  adam(0.0005),  adamgc(0.0005),  esgd(0.01) '''
        optimizer = adamgc


#   Data
#--------------------------------------------------------------------------------------------------
    data_path = 'data/'

    data = mnist(path=data_path, nvalidation=0) 
    #data = mnistBinarized(path=data_path)  # only for UL models

    #data = mnist(path=data_path, distort=3, shuffle=True) 
    #data = freyfaces(path=data_path)

    visualize(-1, data['tr_X'][0:min(len(data['tr_X']),900)], data['shape_x'])


#   Training
#--------------------------------------------------------------------------------------------------
    model = Model(data, hp)
    
    print ("M: %s  lr: %.5f  init: %.1f  batch: %d" % (model.id, learning_rate, init_scale, batch_size)) 
    
    num_iterations = 10000
    freq_save = 20
    freq_sample = 10

    if model.type == 'SL':
        # Supervised learning
        for it in range(num_iterations):
            begin = time.time()
            model.permuteData(data)
            classRateTr = 0.

            for i in xrange(0, len(data['tr_X']) / batch_size):
                predictTrX = model.train(i)
                classRateTr = classRateTr + np.sum(np.argmax(data['tr_Y'][i*batch_size:(i+1)*batch_size], axis=1) == predictTrX)
    
            end = time.time()
            print("%d,%.4f,%.4f,%.2f" % (it, 
                                              1 - classRateTr / len(data['tr_X']), 
                                              1 - np.mean(np.argmax(data['te_Y'], axis=1) == model.test()), 
                                              end - begin))
            # Save model parameters
            if hp.save_model and it % freq_save == 0:
                model.save()

    elif model.type == 'UL':
        # Unsupervised learning
        for it in range(num_iterations):
            begin = time.time()
            model.permuteData()
            
            tr_cost_p = 0.
            tr_cost_q = 0.

            for i in xrange(0, len(data['tr_X']) / batch_size):
                log_p, log_q = model.train(i)
                tr_cost_p += log_p
                tr_cost_q += log_q

            te_cost_p = 0.
            te_cost_q = 0.

            for i in xrange(0, len(data['te_X']) / test_batch_size):
                log_p, log_q = model.testBatch(i)
                te_cost_p += log_p
                te_cost_q += log_q

            #te_cost_p, te_cost_q = model.test()

            end = time.time()
            print("%d,%.2f,%.2f,%.2f,%.2f,%.2f" % (it, 
                                                   tr_cost_q / len(data['tr_X']), 
                                                   te_cost_q / len(data['te_X']), 
                                                   (tr_cost_p + tr_cost_q) / len(data['tr_X']),
                                                   (te_cost_p + te_cost_q) / len(data['te_X']),
                                                   end - begin))
            # Generate samples
            if it % freq_sample == 0:
                y_samples = model.decode(36 * (model.n_t + 1))
                y_samples = np.transpose(y_samples, (1,0,2)).reshape((-1, y_samples.shape[2]))
                visualize(it, y_samples, data['shape_x'])
        
            # Save model parameters
            if hp.save_model and it % freq_save == 0:
                model.save()

