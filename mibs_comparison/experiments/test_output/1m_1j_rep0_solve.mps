NAME          1m_1j_rep0_solve
ROWS
 L  R1
 L  R2
 L  R3
 L  R4
 N  Obj
COLUMNS
    INT1      'MARKER'                 'INTORG'
    YHAT      R4      -1
    YHAT      Obj     -1
    Y11       R2       1
    Y11       R3      -1
    Y11       R4      10
    X1        R1      50
    X1        R2      -1
    X1        R3       1
    INT1END   'MARKER'                 'INTEND'
RHS
    B         R1      100
    B         R2      0
    B         R3      0
    B         R4      0
BOUNDS
 LI BOUND     X1      0
 LI BOUND     Y11     0
 LI BOUND     YHAT    0
ENDATA
