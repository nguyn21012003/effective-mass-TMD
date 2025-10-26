set terminal qt size 900,900


unset key

# set xtics 0.1
set yrange [-0.2:0.2]
unset tics
unset border



set xrange [-10:15]

set arrow from 0, -0.12 to 0, 0.12 nohead lc rgb "#1e1e1e"         lw 2
set arrow from 6.3, -0.12 to 6.3, 0.12 nohead lc rgb "#1e1e1e"     lw 2
set arrow from -6.3, -0.12 to -6.3, 0.12 nohead lc rgb "#1e1e1e"   lw 2
set arrow from 6.3*2, -0.12 to 6.3*2, 0.12 nohead lc rgb "#1e1e1e" lw 2


f1(x) = 0.0041*cos(x) - 0.0509
f2(x) = 0.0041*cos(x + pi) + 0.0509




set arrow from 2.8,0.0544 to 3.15+0.81,0.0544 head filled size screen 0.03,15,45 lc rgb "black" lw 6
set arrow from 2.8,-0.0543 to 3.15+0.81,-0.0543 backhead filled size screen 0.03,15,45 lc rgb "black" lw 6

set arrow from -0.97,-0.01 to -0.97,0.01 head filled size screen 0.03,15,45 lc rgb "black" lw 6
set arrow from 0.95,-0.01 to 0.95,0.01 backhead filled size screen 0.03,15,45 lc rgb "black" lw 6

# set arrow from 5.32,-0.01 to 5.32,0.01 head filled size screen 0.03,15,45 lc rgb "black" lw 6
# set arrow from 7.22,-0.01 to 7.22,0.01 backhead filled size screen 0.03,15,45 lc rgb "black" lw 6

set title "(a) Weak magnetic field" font "CMU Serif, 28" offset 0,-10

plot f1(x) lw 5 lc rgb "black" notitle, \
     f2(x) lw 5 lc rgb "black" notitle, \
     -0.1 w l lw 2 lc rgb "#1e1e1e" notitle, \
      0.1 w l lw 2 lc rgb "#1e1e1e" notitle, \
      '+' u (0):(0):(2):(0.055):(0) every ::::0 w ellipses lw 5 lc rgb "black" notitle, \
      '+' u (6.3):(0):(2):(0.055):(0) every ::::0 w ellipses lw 5 lc rgb "black" notitle, \
      '+' u (6.3*2):(0):(2):(0.055):(0) every ::::0 w ellipses lw 5 lc rgb "black" notitle, \
      '+' u (-6.3):(0):(2):(0.055):(0) every ::::0 w ellipses lw 5 lc rgb "black" notitle, \
      '+' u (3.15):(0):(8.25):(0.111):(0) every ::::0 w ellipses lw 2 dt 7 lc rgb "black" notitle

