function [outFile] = ps5_plot(one, two, four, five, oneE, twoE,fourE,fiveE)

v1 = xlsread(one, 'A1:A100000');
pd1 =  xlsread(one, 'B1:B100000');
pm1 = xlsread(one,  'C1:C100000');
true =  xlsread(four, 'D1:D100000');


v2 = xlsread(two, 'A1:A100000');
pd2 =  xlsread(two, 'B1:B100000');
pm2 = xlsread(two,  'C1:C100000');

v4 = xlsread(four, 'A1:A100000');
pd4 =  xlsread(four, 'B1:B100000');
pm4 = xlsread(four,  'C1:C100000');

v5 = xlsread(five, 'A1:A100000');
pd5 =  xlsread(five, 'B1:B100000');
pm5 = xlsread(five,  'C1:C100000');



v1E = xlsread(oneE, 'A1:A100000');
pd1E =  xlsread(oneE, 'B1:B100000');
pm1E = xlsread(oneE,  'C1:C100000');
true =  xlsread(four, 'D1:D100000');


v2E = xlsread(twoE, 'A1:A100000');
pd2E =  xlsread(twoE, 'B1:B100000');
pm2E = xlsread(twoE,  'C1:C100000');

v4E = xlsread(fourE, 'A1:A100000');
pd4E =  xlsread(fourE, 'B1:B100000');
pm4E = xlsread(fourE,  'C1:C100000');

v5E = xlsread(fiveE, 'A1:A100000');
pd5E =  xlsread(fiveE, 'B1:B100000');
pm5E = xlsread(fiveE,  'C1:C100000');
t = linspace(1,100000, 100000)';

figure;
hold on
legend('truth', 'Viterbi','decoding','mean')
title('Initial Decodings, 1mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v1,'g-', t,pd1,'b-', t,pm1, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Initial Decodings, 2mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v2,'g-',t,pd2,'b-', t,pm2, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Initial Decodings, 4mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v4,'g-', t,pd4,'b-', t,pm4, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Initial Decodings, 5mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v5,'g-', t,pd5,'b-', t,pm5, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Initial Decodings, 1mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v1E,'g-', t,pd1E,'b-', t,pm1E, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Estimated Decodings, 2mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v2E,'g-', t,pd2E,'b-', t,pm2E, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')

hold off

figure;
hold on
title('Estimated Decodings, 4mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v4E,'g-', t,pd4E,'b-', t,pm4E, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off

figure;
hold on
title('Estimated Decodings, 5mu')
xlabel('locus')
ylabel('TMRCA')
plot(t,v5E,'g-', t,pd5E,'b-', t,pm5E, 'r-', t,true,'k-')
legend('Viterbi', 'decoding','mean','truth')
legend('Location', 'North')
hold off
 
end