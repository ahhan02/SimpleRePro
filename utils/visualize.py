import visdom
import time
import numpy as np

class Visualizer(object):
   '''
   @description: you can still call `self.vis.function` or `self.function` as follows.
        self.text('hello visdom')
        self.histogram(t.randn(1000))
        self.line(t.arange(0, 10),t.arange(1, 11))
   @param : undefined
   @return: 
   ''' 

   def __init__(self, env='default', port=8097, **kwargs):
       self.vis = visdom.Visdom(env=env, port=port, **kwargs)
       self.index = {} 
       self.log_text = ''

   def reinit(self, env='default', **kwargs):
       self.vis = visdom.Visdom(env=env, **kwargs)
       return self

   def plot_many(self, d):
       '''
       @description: 
       @param : d(dict): (name, value) i.e. ('loss', 0.11)
       @return: 
       '''
       for k, v in d.iteritems():
           self.plot(k, v)

   def img_many(self, d):
       for k, v in d.iteritems():
           self.img(k, v)

   def plot(self, name, y, **kwargs):
       '''
       @description: self.plot('loss', 1.00)
       @param : undefined
       @return: 
       '''
       x = self.index.get(name, 0)
       self.vis.line(Y=np.array([y]), X=np.array([x]),
                     win=name,
                     opts=dict(title=name),
                     update='append' if x > 0 else None,
                     **kwargs)
       self.index[name] = x + 1

   def img(self, name, img_, **kwargs):
       '''
       @description: 
            self.img('input_img',  t.Tensor(64, 64))
            self.img('input_imgs', t.Tensor(3, 64, 64))
            self.img('input_imgs', t.Tensor(100, 1, 64, 64))
            self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
       @param : undefined
       @return: 
       '''
       self.vis.images(img_.cpu().numpy(),
                      win=name,
                      opts=dict(title=name),
                      **kwargs)

   def log(self, info, win='log_text'):
       '''
       @description: self.log({'loss':1, 'lr':0.0001})
       @param : undefined
       @return: 
       '''
       self.log_text += ('[{time}] {info} <br>'.format(
                           time=time.strftime('%m%d_%H%M%S'),
                           info=info))
       self.vis.text(self.log_text, win)

   def __getattr__(self, name):
       '''
       @description: `self.function` is same as `self.vis.function` except for `plot`, `image`, `log`, `plot_many`
       @param : undefined
       @return: 
       '''
       return getattr(self.vis, name)