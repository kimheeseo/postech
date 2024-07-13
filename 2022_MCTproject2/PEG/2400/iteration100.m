x=[0 0.5 1.0 1.5 2.0 2.5 3.0 3.5];
y1=[0.2832 0.266 0.2557 0.2416 0.2253 0.2068 0.191 0.1603]
y2=[0.2823 0.2663 0.2554 0.241 0.2256 0.2057 0.1898 0.1625]

plot(x,y1,'b--o')
hold on
plot(x,y2,'r--o')

legend('gallaer','PEG')
hold off

title('iteration 100, legnth 2400')
xlabel('Eb/No(dB)')
ylabel('BER')