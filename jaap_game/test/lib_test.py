import pygame as pg

pg.font.init()
f = pg.font.SysFont('segoeui', 30)

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 

def yell():
  print("AAAAAAAAAAAAAAAAH")

class noise(object):
  def __init__(self, text):
    self.text = text

  def yell(self):
    print(self.text)

  def whisper(self, text):
    print(text)

steps = 0

def step():
  pg.time.delay(500)
  global steps
  steps += 1
  pg.draw.rect(screen, (1,255,1), [0, 0, 300, 300])
  
  text = "steps: " + str(steps)
  screen.blit(f.render(text, 1, pg.color.THECOLORS["white"]),(150,120))

  pg.display.flip()