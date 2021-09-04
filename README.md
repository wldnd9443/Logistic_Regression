# # Logistic Regression
> Logistic Regression 구현해보기

앞서 배웠던 Linear Regression과 달리 Logistic Regression은 어떤 방법으로 학습하는지 그래프와 그림을 통해 확인합니다.


![](../header.png)

## 소개

  이 Repository에서는 Logistic Regression을 구현해 볼 것입니다. 이전에 다뤘던 Linear Regression과 마찬가지로 Logistic Regression도 기존에 주어진 데이터를 대표할 수 있는 그래프를 찾아야 합니다. 그러나 Linear Regression과 달리 Logistic Regression은 데이터를 True와 False로 구분하는 classification 입니다. 학습을 완료하고 입력에 따른 결과값이 Linear와 다르게 True 나 False를  반환해야 합니다.    
  
## 이론 
Linaer Regression은 데이터를 대표할 수 있는 직선 그래프를 찾고 이 그래프를 통해 추후에 입력값을 그래프를 통해 예측하는 방식이었습니다. 하지만 True와 False로 이루어진 Data Set에서는 직선이 이 데이터를 대표할 수는 없습니다.

![linear_binary](https://user-images.githubusercontent.com/44831709/131362869-9e7e9996-ac7b-49bc-8985-62f7c9e42e80.png)   

이 때문에 현재의 데이터 셋을 대표할 수 있는 그래프를 고안해냅니다. 이 그래프는 Sigmoid 함수 입니다.

![sigmoid_function](https://user-images.githubusercontent.com/44831709/132097048-b6b14a63-a387-4a3d-ac72-d24be17bd386.png)

Sigmoid함수는 기본적으로 아래와 같은 모습을 합니다. 
![Hypothesis](https://user-images.githubusercontent.com/44831709/132097332-b4ecc5fb-c60d-4481-8980-fe9590a1d9fa.png)
![sigmoid_default](https://user-images.githubusercontent.com/44831709/132097256-5903b99c-ecf9-4588-ad49-98cfdf9066e4.png)

이 함수는 beta1과 beta0값을 수정하면서 아래와 같이 형태를 바꿀 수 있습니다. 

![sig_beta1](https://user-images.githubusercontent.com/44831709/132101508-0f5065eb-b4f3-4b31-b25d-db74023c0479.png)
![sig_beta0](https://user-images.githubusercontent.com/44831709/132101519-0ff4571f-1d37-49e5-8115-03c9c282dfff.png)




## 구현 과정

두 가지 경향성이 있는 임의의 2차원를 설정합니다.

```
mat_covs = np.array([[[1,0.9],[0.9,1]],[[1,0.9],[0.9,1]]])
# mat_covs = np.array([[[1,0.8],[0.8,1]],[[1,0.8],[0.8,1]]])

mus =  np.array([[4,3],[2,3]])
Ns = np.array([400,400])

X = np.zeros((0,mus.shape[1]))
Y = np.zeros(0)

cls = 0
for mu,mat_cov,N in zip(mus, mat_covs, Ns):
    X_ = np.random.multivariate_normal(mu, mat_cov, N)
    Y_ = np.ones(N)*cls
    X = np.vstack((X,X_))
    Y = np.hstack((Y,Y_))
    cls += 1
    
cls_unique = np.unique(Y)

```

Cost function은 아래와 같이 MSE(Mean Squared Error)로 설정합니다.   


![cost_function](https://user-images.githubusercontent.com/44831709/130806508-eae6ef66-e175-4f52-acbf-edba20e9aa6f.png)


데이터를 학습니다. history 에 베타값을 저장하고 나중에 그래프를 그릴 때 활용합니다.

```
history = []
history.append([beta_0,beta_1])
for i in range(N_iter):
    beta_1_next = beta_1 - alpha * (1/N) * np.sum(2*beta_1*np.square(x1) -2*x1*x2 + 2*x1*beta_0)
    beta_0_next = beta_0 - alpha* (1/N) * np.sum(2*beta_0 - 2*x2 + 2*x1*beta_1)
    beta_1, beta_0 = beta_1_next, beta_0_next
    history.append([beta_0,beta_1])

```



## 정보

https://wikidocs.net/21670
