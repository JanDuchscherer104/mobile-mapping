Die gegebenen Orientierungen wurden als 3x3 Matrizen R gespeichert. Aufgrund von Rundungsfehlern werden 
diese nicht mehr die Eighenschaften einer Rotation (also R'=inv(R) und det(R)=1) haben. 
Ähnlich wie bei der Berechnung der Fundamentalmatrix kann jetzt die Eigenschaft einer Rotation aber wieder mit Hilfe der Singulärwertzerlegung 
erzwungen werden, wie folgt:

R = U*D*V'  (*) 

Setzte also

D_new(1,1) = 1
D_new(2,2) = 1
D_new(3,3) = det(U*V')   (**)

und insgesamt

R = U*D_new*V'

(*) (bei einer Rotation muss für die drei Diagonalelemente gelten D(i,i) = abs(1))  
(**) U*V' ist entweder +1 oder -1; so wird erreicht, dass die Determinante insgesamt wieder +1 ergibt
 
 