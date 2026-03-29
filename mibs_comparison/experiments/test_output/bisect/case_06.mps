NAME          case_06
ROWS
 L  BUDGET
 L  ASSIGN_UP
 L  ASSIGN_LO
 L  MAKESPAN
 N  OBJ
COLUMNS
    YHAT       OBJ        -1
    YHAT       MAKESPAN   -1
    Y11        ASSIGN_UP  1
    Y11        ASSIGN_LO  -1
    Y11        MAKESPAN   10
    INT1      'MARKER'                 'INTORG'
    X1         BUDGET     50
    X1         ASSIGN_UP  -1
    X1         ASSIGN_LO  1
    INT1END   'MARKER'                 'INTEND'
RHS
    B         BUDGET     100
    B         ASSIGN_UP  0
    B         ASSIGN_LO  0
    B         MAKESPAN   0
BOUNDS
 LI BOUND     X1         0
 LO BOUND     YHAT       0
 LO BOUND     Y11        0
ENDATA
