library(xlsx)
library(ggplot2)
path<-"F:/Works/Summer Research/Reports/Simulation3 (meeting 6)/Data/plt_data.xlsx"
pltdata<-read.xlsx(path,1)
ggplot(pltdata,aes(x=mean))+
  geom_line(aes(y=E1,color="G1"))+
  geom_line(aes(y=E2,color="G2"))+
  theme_bw(base_family = "Times")+
  scale_fill_discrete(labels=c("Theoretical","Bootstrap"))+
  scale_color_discrete(labels=c("Theoretical","Bootstrap"))+
  theme(panel.grid = element_blank(),
        legend.position = "top",                      # legend 置顶
        panel.border = element_blank(),
        text = element_text(family = "STHeiti"),      # Mac 系统中中文绘图
        plot.title = element_text(hjust = 0.5)) +     # 标题居中
  labs(x = 'mean', y = expression(C[min]), title = ("Expectation(n=100),theta2=1"),
       color = "", fill = "")
