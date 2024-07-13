x=[0 0.5 1.0 1.5 2.0 2.5 3.0 3.5];
y3=[0.2826 0.2698 0.2522 0.2444 0.2242 0.2066 0.1893 0.1533]
y4=[0.2845 0.2705 0.2544 0.2436 0.2259 0.2027 0.1884 0.1512]

plot(x,y3,'b--o')
hold on
plot(x,y4,'r--o')
legend('gallaer', 'PEG')
hold off

title('iteration 20, legnth 600')
xlabel('Eb/No(dB)')
ylabel('BER')
