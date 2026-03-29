NAME          1m_1j_rep0
ROWS
 N  OBJ
 L  BUDGET
 L  ASSIGN_UP_1
 L  ASSIGN_LO_1
 L  MAKESPAN_1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x_1        OBJ        0.0
    x_1        BUDGET     50
    x_1        ASSIGN_UP_1   -1.0
    x_1        ASSIGN_LO_1   1.0
    x_1        MAKESPAN_1   0.0
    y_hat      OBJ        -1.0
    y_hat      BUDGET     0.0
    y_hat      ASSIGN_UP_1   0.0
    y_hat      ASSIGN_LO_1   0.0
    y_hat      MAKESPAN_1   -1.0
    y_1_1      OBJ        0.0
    y_1_1      BUDGET     0.0
    y_1_1      ASSIGN_UP_1   1.0
    y_1_1      ASSIGN_LO_1   -1.0
    y_1_1      MAKESPAN_1   10
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS        BUDGET     100
    RHS        ASSIGN_UP_1   0.0
    RHS        ASSIGN_LO_1   0.0
    RHS        MAKESPAN_1   0.0
BOUNDS
 LI BOUND      x_1       0
 LI BOUND      y_hat      0
 LI BOUND      y_1_1     0
ENDATA
