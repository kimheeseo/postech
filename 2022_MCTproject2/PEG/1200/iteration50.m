x=[0 0.5 1.0 1.5 2.0 2.5 3.0 3.5];
y2=[0.2831 0.2685 0.2552 0.2392 0.2254 0.2076 0.187 0.1582]
y1=[0.2845 0.2691 0.2543 0.2413 0.2274 0.2095 0.186 0.164]

plot(x,y1,'b--o')
hold on
plot(x,y2,'r--o')

legend('gallaer', 'PEG')
hold off

title('iteration 50, legnth 1200')
xlabel('Eb/No(dB)')
ylabel('BER')
