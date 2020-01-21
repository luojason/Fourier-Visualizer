import graphics as g
import math
import cmath

#parameters to modify framerate, period of the function, etc
dtheta = .01 #theta displacement, in radians
sample_threshold = 200

def c2p(c):
    """Creates a Point object corresponding to the given complex number."""
    return g.Point(c.real, c.imag)

def p2c(p):
    """Returns the complex number corresponding to a given Point object."""
    return p.getX() + p.getY()*1j

def interpolate_set(pt_list, threshold):
    """Performs linear interpolation on a set to increase sample size to the given threshold."""
    sample_size = len(pt_list)
    assert sample_size > 0
    if sample_size >= threshold: return pt_list
    upsample_factor = math.ceil(threshold/sample_size)
    coord_list = []
    for i in range(sample_size):
        current_pt, next_pt = p2c(pt_list[i]), p2c(pt_list[(i + 1)%sample_size])
        step = (next_pt - current_pt)/upsample_factor
        coord_list.append(current_pt)
        for num_rep in range(upsample_factor - 1):
            current_pt += step
            coord_list.append(current_pt)
    return [c2p(coord) for coord in coord_list]

def dft_inefficient(pt_list):
    """Calculates the discrete fourier transform on the list of samples.
    Inefficient implementation; directly calculates from the formula. O(n^2) algorithm."""
    dft = []
    N = len(pt_list)
    for k in range(N):
        coeff = 0
        for n,x in enumerate(pt_list):
            root = cmath.exp(-2*cmath.pi*k*n*1j/N)
            coeff += p2c(x)*root
        dft.append(coeff/N)
    return dft

class DFT_Renderer:
    """Wrapper class for dft and display operations"""

    #display configuration options
    line_width = 1
    line_color = 'gray'
    line_arrow = 'none'
    radius = 2
    circ_color = 'red'
    trail_length = 50
    res_step = 1 #number of terms to add to the series when inc is called
    
    def line_config(p1, p2):
        """Generates a Line object. Desired options are configurated in this method.
        Used by DFT_Renderer to configure series line/vector display."""
        l = g.Line(p1, p2)
        l.setWidth(DFT_Renderer.line_width)
        l.setOutline(DFT_Renderer.line_color)
        l.setArrow(DFT_Renderer.line_arrow)
        return l

    def circle_config(p1):
        """Generates a list of Circle objects. Desired options are configurated in this method.
        Used by DFT_Renderer to configure series evaluation display."""
        circles = [g.Circle(p1, DFT_Renderer.radius) for i in range(DFT_Renderer.trail_length)]
        for circ in circles:
            circ.setOutline(DFT_Renderer.circ_color)
            circ.setFill(DFT_Renderer.circ_color)
        return circles
    
    def __init__(self, pt_list, dtheta, win):
        assert len(pt_list) > 0
        self.dtheta = dtheta
        self.win = win
        #metadata initialization
        self.theta = 0
        self.dft = dft_inefficient(pt_list)
        self.lines = []
        self.loc_pos = None
        self.loc_circles = None
        self.trail_num = 0
        self.num_vec = 10

    def undisplay(self):
        """Removes all figures from the current window."""
        for line in self.lines: line.undraw()

    def display(self):
        """Draws all figures onto the given window.
        Initialized trails of circles if not done so already."""
        for line in self.lines: line.draw(self.win)
        if self.loc_circles == None:
            self.loc_circles = DFT_Renderer.circle_config(c2p(self.loc_pos))
            for circ in self.loc_circles: circ.draw(self.win)
        current_circle = self.loc_circles[self.trail_num]
        old_center = current_circle.getCenter()
        shift = self.loc_pos - p2c(old_center)
        current_circle.move(shift.real, shift.imag)
        self.trail_num = (self.trail_num + 1)%DFT_Renderer.trail_length

    def update(self):
        """Updates all figures."""
        N = len(self.dft)
        del self.lines[:]
        current_loc = 0
        prev_p = c2p(current_loc)
        current_loc += self.dft[0]
        current_p = c2p(current_loc)
        self.lines.append(DFT_Renderer.line_config(prev_p, current_p))
        for n in range(1, self.num_vec):
            #adding nth term of series
            prev_p = current_p
            current_loc += self.dft[n%N]*cmath.exp(n*self.theta*1j)
            current_p = c2p(current_loc)
            self.lines.append(DFT_Renderer.line_config(prev_p, current_p))
            #adding -nth term of series
            prev_p = current_p
            current_loc += self.dft[-n%N]*cmath.exp(-n*self.theta*1j)
            current_p = c2p(current_loc)
            self.lines.append(DFT_Renderer.line_config(prev_p, current_p))
        self.loc_pos = current_loc
        self.theta += self.dtheta

    def inc_terms(self):
        self.num_vec += DFT_Renderer.res_step

    def dec_terms(self):
        if self.num_vec > DFT_Renderer.res_step: self.num_vec -= DFT_Renderer.res_step

if __name__ == '__main__':
    win = g.GraphWin('Fourier Series Visualizer', 800, 800, autoflush=False)
    win.setCoords(-500, -500, 500, 500)
    info = g.Text(g.Point(0,470), 'Click to set a vertex. Press any button and click to finish.')
    info.draw(win)
    pt_list = []
    poly = g.Polygon(pt_list)
    while True: #grabbing sample points
        pt = win.getMouse()
        if win.checkKey() != '': break
        pt_list.append(pt)
        poly.undraw()
        poly = g.Polygon(pt_list)
        poly.draw(win)
    upsampled_pts = interpolate_set(pt_list, sample_threshold)
    dft = DFT_Renderer(upsampled_pts, dtheta, win)
    info.setText("Press ' up' or ' down' to increase/decrease the number of terms in the series. Press ' q' to quit.")
    while True: #running FS animation attempt
        dft.undisplay()
        dft.update()
        dft.display()
        g.update(30)
        key = win.checkKey()
        if key == 'Up': dft.inc_terms()
        elif key == 'Down': dft.dec_terms()
        elif key == 'q': break
    win.close()
