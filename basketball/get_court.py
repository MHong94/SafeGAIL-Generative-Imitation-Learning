import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc


def draw_half_court_left(ax, color='black', lw=1):
    # Create various parts of an NBA basketball court
    # Court boundaries
    # ax.set_xlim(0, 46.998)
    # ax.set_ylim(0, 50)

    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    boundaries = Rectangle((0, 0), 46.998, 50, linewidth=lw, color=color, fill=False)
    ax.add_patch(boundaries)

    # The paint
    outer_box_left = Rectangle((0, 17), 19, 16, linewidth=lw, color=color, fill=False)
    inner_box_left = Rectangle((0, 19), 19, 12, linewidth=lw, color=color, fill=False)
    ax.add_patch(outer_box_left)
    ax.add_patch(inner_box_left)

    # Free Throw Lines
    top_free_throw_left = Arc((19, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    bottom_free_throw_left = Arc((19, 25), 12, 12, theta1=90, theta2=-90, linewidth=lw, color=color, fill=False,
                                 linestyle='dashed')
    ax.add_patch(top_free_throw_left)
    ax.add_patch(bottom_free_throw_left)

    # Three-point lines
    top_three_left = Rectangle((0, 47), 14, 0, linewidth=lw, color=color, fill=False)
    bottom_three_left = Rectangle((0, 3), 14, 0, linewidth=lw, color=color, fill=False)
    three_arc_left = Arc((5.2493, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, linewidth=lw, color=color, fill=False)
    ax.add_patch(top_three_left)
    ax.add_patch(bottom_three_left)
    ax.add_patch(three_arc_left)

    # Backboard and hoops
    hoop_left = Circle((5.2493, 25), 0.75, linewidth=lw, color=color, fill=False)
    backboard_left = Rectangle((4, 22), 0, 6, linewidth=lw, color=color, fill=False)
    ax.add_patch(hoop_left)
    ax.add_patch(backboard_left)

    # Restricted Area
    restricted_left = Arc((5.2493, 25), 8, 8, theta1=-90, theta2=90, linewidth=lw, color=color, fill=False)
    ax.add_patch(restricted_left)
    return ax


fig = plt.figure()
ax = draw_half_court_left(ax=plt.gca())
# plt.xlim(-300, 300)
# plt.ylim(-100, 500)
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('court_o.png', bbox_inches=extent)
plt.show()
