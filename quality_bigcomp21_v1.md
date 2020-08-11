---
abstract: |
  Real-time bidding has become one of the most important forms of
  advertising and already grows a billion-dollar business. While
  real-time bidding brings a huge profit for online business, it also
  becomes a potential target not only for malicious purposes, e.g.
  bid-requests from botnet, hired user, etc. but also the other such as
  crawlers, users. In display advertising, bid-request traffics was
  classified into two kinds: intentional and non-intentional. From a
  perspective of demand-side platform (DSP) in the display advertising
  model, the budget of advertisers should be used as effectively as
  possible by limiting the intentional traffics. Therefore, we want to
  classify and predict these two kinds of bid-request traffic. However,
  since the behaviors of intentional bid-requests are possible to define
  while the non-intentional bid-requests are not, it is a challenge for
  display advertising to evaluate inventory quality of these traffics,
  especially in DSP side. Moreover, we cannot directly apply the
  traditional techniques of machine learning to classify these kinds of
  traffics. In this paper, we introduce a novel of evaluating inventory
  quality on Demand-Side in display advertising and present two
  approaches based on Positive and Unlabeled learning to classify the
  bid-request traffics. In particular, we first preprocess the
  bid-request dataset to define the abnormal behaviors and label
  bid-requests into intentional and unlabeled. Then we will apply two PU
  learning methods to build classifier models. This study provides some
  important insights into the practice of ad fraud detection for DSP
  side in display advertising.
author:
- |
  \*\
bibliography:
- quality_bigcomp21.bib
title: 'Non-Intentional Inventory Detection on Demand-Side in Display
  Advertising'
---

Display advertising, non-intentional inventory, semi-supervised learning

Introduction
============

With the growth of ad spending in display adverting, display advertising
plays a pivotal role in digital advertising industry since advertisers
and publishers are easy to distribute and to place their ads on their
websites and applications. Audience, advertiser and publisher are three
main roles in display advertising. When an audience visits publisher's
websites or uses publisher's applications, it triggers publisher's
scripts to send an ad request to connected advertisers for ad
impression. The advertiser responses suitable ad for impression based on
the ad request, which contains the information of audience and publisher
[@13_ADKDD_Yuan].

The Demand-Side Platform (DSP) helps advertisers manage their campaigns
and maximize the Key Performance Indicator (KPI) with a given budget in
a period for each campaign, where the KPI could be impression, view,
click, conversion and so on [@14_KDD_Zhang]. Given the KPI of a campaign
is click, DSP tries to make more clicks as possible subject to the
budget and the period of the campaign. In other words, DSP tries to
minimize the Cost Per Click (CPC)[^1] as well as consume the budget for
the campaign. On the other hand, the publishers supply their inventory,
the ad requests as well as the ad impression opportunities, to
demand-side for the revenue. When DSP receives an ad request, it will
find all eligible ads from all campaigns and try to select the best one
for impression. The payment is decided by auction [@14_KDD_Zhang] or by
contract [@12_KDD_Bharadwaj][@18_KDD_Jauvion]. In general, the selection
strategy considers the possibility of audience achieving the
corresponding KPI as well as the cost to publisher if the ad is
successfully rendered on audience's browser or application.

The KPI of campaign represents how advertiser expects audience to
interact with their ads. However, some of inventory are from
non-intentional sources [@13_KDD_Stitelman]. For example, the crawler is
disguised as a normal browser to avoid being banned by publisher's
server, but it triggers the ad request to demand-side. Even fraudulent
activities are generated to cheat advertisers of their advertising
budget. For advertisers, they do not want to deliver their ads for
non-intentional inventory since it not only wastes the advertising
budget campaign but also affects the campaign performance. Thus, for
DSP, the non-intentional inventory should be filtered out in the
beginning and be not considered for ad delivery.

To identify non-intentional inventory, it is hard to find a few key
common rules to discover them, especially the fraudulent activities,
which are cunningly manufactured to grab the advertiser's budget
[@11_IMC_Stone-Gross]. In general, publishers allocate their inventory
to different channels to seek more revenue. It causes that DSP only
receives segments of inventory from a source intermittently. Even the
regular visiting patterns of some crawlers are affected in DSP.
Furthermore, there is no ground truth for labeling the non-intentional
inventory. The above challenges greatly increase the difficulties of
battling non-intentional inventory, especially on demand-side.

In this paper, we treat non-intentional inventory identification as a
semi-supervised learning problem since it is difficult to label all
intentional inventory or non-intentional inventory. In particular, we
could identify some non-human activities since the activity behavior is
like non-human generated[^2]. Then we could label the corresponding
inventory as non-intentional. In contrast, it is hard to find some rules
to ensure that the inventory is intentional. Thus, we are able to label
partially non-intentional inventory only. This kind of semi-supervised
learning problem is Positive-Unlabeled (PU) learning [@02_KDD_Yu]. For
our problem, the positive label represents non-intentional inventory,
and other non-labeled could be non-intentional or intentional. To
distinguish non-intentional activities and intentional activities, our
approach is to map the inventory activities into feature space for each
inventory, and then to identify the non-intentional inventory from the
unlabeled inventory based on the similarity to labeled non-intentional
inventory. For the variable activities of non-intentional inventory, our
concept is to extract the representative features from the behavior
behind inventory via feature engineering.

In summary, our contributions are outlined as follows:

-   We formulate the non-intentional inventory identification as a PU
    learning problem.

-   We utilize feature engineering techniques to extract the access
    behavior of inventory.

-   We conduct a comprehensive evaluation on real datasets.

The remainder of this paper is organized as follows: Section
[2](#sec:related){reference-type="ref" reference="sec:related"}
discusses the related works. Section
[3](#sec:problem){reference-type="ref" reference="sec:problem"} and
[4](#sec:method){reference-type="ref" reference="sec:method"} present
the problem definition and our approach details, respectively. Section
[5](#sec:evaluation){reference-type="ref" reference="sec:evaluation"}
shows the experimental results of our approach. Lastly, Section
[6](#sec:conclusion){reference-type="ref" reference="sec:conclusion"}
concludes this paper.

Related Work {#sec:related}
============

In display advertising, advertisers would like to interact with real
users, and would not like to waste their budgets on non-intentional
inventory. The recent researches of non-intentional traffics fall under
two classes: (1) non-frauds, which trigger the ad request to demand-side
and disguised as a normal user such as crawler[@02_ICCCN_Yuan] ; (2)
frauds, which come from various sources such as
botnet[@11_IMC_Stone-Gross][@12_INFOCOM_Soldo], click
spam[@12_SIGCOMM_Dave], crowdsourcing[@12_WWW_Wang], sophisticated
sources triggered by real
user[@13_KDD_Stitelman][@13_USS_Springborn][@19_WWW_Pastor]. To deal
with non-fraud of non-intentional traffics, [@02_ICCCN_Yuan] has
proposed an active indexing system to eliminate Web crawlers. This
approach used an active network to remove unnecessary crawler traffic by
constantly monitoring and analyzing traffics on strategic routers. For
the fraudulent activities in display advertising, [@11_IMC_Stone-Gross]
introduced the first large-scale fraudulent activities in online ad
exchange by analyzing the ingress and egress ad traffics and examing
information from command-and-control botnet used for ad fraud. Moreover,
this paper also tried a number of models that are relatively simple to
explore methods that might be able to identify suspicious click traffic
and used ground truth of good or bad publishers to evaluate them.
Considering the impact of botnet, [@12_INFOCOM_Soldo] has built a
detecting machine-generated traffic framework based on the IP size
information to detect inflation attacks. This framework deployed
statistical learning techniques and ensemble learning to detect and
classify the anomalous deviation from the expected publisher's IP size
distribution. Meanwhile, Click spam is a kind of fraud that can be
generated using a variety of approaches, which are: i) botnet; ii)
tricking or confusing users into clicking ads; iii) directly paying
users to click on ads. With attention to the click-spam problem in ad
networks, [@12_SIGCOMM_Dave] introduced the first approach for
advertisers to measure click-spam rates on their ad and developed an
automated methodology for ad networks to identify simultaneous
click-spam attacks. In this estimation method, the authors use a
Bayesian approach to deal with the lack of ground truth. Furthermore,
the method can only estimate of the proportion of click-spam presenting
in the click traffic, it cannot identify specifically whether a click is
a click-spam or not. To better understanding malicious crowd-sourcing
systems, a crowdturfing systems have been studied comprehensively in
[@12_WWW_Wang]. These systems work as crowd-sourcing systems, e.g.
Amazon's Mechanical Turk. Nonetheless, they allow creating unlawful or
unethical tasks like mass account creation or posting of particular
content on online social networks, blogs, and online forums. The authors
used detailed crawls to estimate the size and operational structure of
these crowdturfing systems. They also created benign campaigns to assess
the effectiveness of these systems. They report that \$4 million dollars
have been spent on only the two largest crowdturfing sites and the
number of campaigns on these two sites is increasing rapidly. The
results also show that campaigns on crowdturfing systems are effective
at attracting real user responses. Due to the increasing fraudulent
traffics to websites, a real-time approach for classifying and filtering
non-intentional traffics from weblogs was introduced by
[@13_KDD_Stitelman]. This approach consists of two stages: i)
Co-visitation network was proposed to identify malicious sites which
have a weirdly large amount of browsers access overlap; ii) the
second-step classification method called \"penalty box\" was used to
filter non-intentional traffics. Likewise, invalid traffic generated by
pay-per-view (PPV) networks has been thoroughly studied in
[@13_USS_Springborn]. The authors got an insight into PPV networks by
analyzing the purchased traffic for their honeypot websites. The traffic
was bought from popular PPV network service providers. To generate
traffic for the target website, the providers pay legitimate publishers
for embedding their tags on the publishers' websites. Then, if these
websites are visited, pop-under windows will be created to load the
target website silently without the awareness of the user. The research
shows that hundreds of millions of dollars have been lost for PPV
networks annually. Similarly, [@19_WWW_Pastor] has built a system for
the detection of invalid ad traffic in real-time at the level of
individual requests called Nameles that satisfy the requirements of both
advertisers and DSPs. The abnormal ad requests in this paper can be
identified by computing anomaly score of the distribution of bid
requests across IP addresses in each domain based on a normalized
version of Shannon entropy. The difference from most of the mentioned
related works is that our approach is a data mining approach with the
data collected at the DSP side and we try to identify non-intentional
traffic at the bid-request level. [@19_WWW_Pastor] is the most similar
work with us. However, our work is the first try to model the problem as
PU learning problem.

PU learning is known as a special case of semi-supervised learning which
has been proposed to deal with the setting when only positive and
unlabeled data available. Due to the characteristic of PU learning, it
is effective in various real-world situations such as text
classification[@03_ICML_Lee], web page classification[@02_KDD_Yu],
anomaly detection[@18_WWW_Zhang]. For the purpose of web page
classification to achieve the accuracy as high as traditional SVM (full
labeled data), [@02_KDD_Yu] presents PEBL framework based on
Mapping-Convergence (M-C) algorithm which includes two stages. In the
first stage, the mapping stage, a weak classifier is used to identify
strong negative examples. Then, in the second stage, the convergence
stage, the algorithm runs a second classifier repeatedly to make a
gradually finer approximation of negative data. The second classifier
must have the ability to maximize the margin. In an investigation into
text classification, [@03_ICML_Lee] presented the problem of using
weighted logistic regression to learn from positive and unlabeled
examples. In detail, this study used the real value output by applying
logistic regression on weighted examples for learning with positive and
unlabeled examples. It also introduces a performance measure to evaluate
retrieval performance. The performance measure can be estimated from
positive and unlabeled examples. The result shows that the proposed
methods are effective on a text classification corpus. Additionally,
considering the performance of PU learning with the availability of
large unlabeled examples, [@14_PRL_Mordelet] introduced a new method for
PU learning based on bagging techniques, called bagging SVM. There are
four main steps to this method. First, a training set is generated by
associating positive examples with a random subsample from unlabeled
examples. Second, a classifier was built from the bootstrap sample by
considering positive examples as positive and unlabeled examples as
negatives. Third, a classifier was trained to the unlabeled examples
that were not included in random subsamples and update scores. Finally,
all these three step are repeated many times and averages their
predictions. The study of the anomaly detection in PU learning was
presented by [@18_WWW_Zhang]. It proposed Anomaly Detection with partial
Observed Anomalies method based on two-stage manner of PU learning. The
method first gathers observed anomalies into different clusters and
selected unlabeled samples to find potential anomalies and reliable
normal samples. Then, they attached a weight to each sample and build a
weighted multi-classification model to discriminate anomalies from
normal samples.

Preliminaries {#sec:problem}
=============

In this section, we give a formal definition of non-intentional
inventory detection problem. First, we define the unit of inventory, ad
request, as follows:

::: {#def:ad_request .definition}
**Definition 1**. *(Ad request) An ad request $\mathbf{x}$ is sent from
publisher to ask for an ad to be impressed. $\mathbf{x}$ contains the
information of audience and information of publisher. In this paper,
$\mathbf{x}$ is represented as a vector with size $m$, where
$\mathbf{x} = [x_1, x_2, ..., x_m]^T \in \mathbb{R}^m$.*
:::

Then, we define non-intentional inventory detection problem as follows:

::: {.definition}
**Definition 2**. *(Non-intentional inventory detection) Given a set of
ad requests labeled as non-intentional $D_P$ and a set of unlabeled ad
requests $D_U$, the problem is to build a function $f(\cdot)$ to
classify an unlabeled ad request $\mathbf{x} \in D_U$ as non-intentional
or intentional. $y=f(\mathbf{x})$ is the classified result of
$\mathbf{x}$. $y=1$ and $y=0$ denote non-intentional and intentional,
respectively.*
:::

Our Proposed Approach {#sec:method}
=====================

Framework Overview
------------------

Figure [\[fig:framework\]](#fig:framework){reference-type="ref"
reference="fig:framework"} shows our framework for non-intentional
detection in this paper. Two parts are in our detection framework,
offline and online. In offline, the goal is to build a model for
detection based on historical data. In online, the goal is to classify
each incoming ad request into intentional or non-intentional based on
the classifier built in offline. If the incoming ad request is detected
as non-intentional, it will be dropped directly; otherwise, ads from
eligible campaigns will be considered to response for delivering. We
show the stage details in the following subsections.

Feature Engineering
-------------------

In feature engineering stage, each raw ad request will be translated to
a real number vector with size $m$ defined in Definition
[Definition 1](#def:ad_request){reference-type="ref"
reference="def:ad_request"}. The raw ad request consists of the
information of publisher (such as, URL, app name, ad type[^3], ad size)
and the information of audience (such as, age, gender, detected
interests) provided from publisher. An example is the bid request
defined by OpenRTB[^4].

In this paper, for an incoming ad request, we focus on extracting the
features that represent the access behavior associated with the incoming
ad request since the access behavior is highly related to the intention
behind the incoming ad request [@14_JMLR_Oentaryo][@02_ICCCN_Yuan]. To
represent the access behavior of the incoming ad request, we describe
the access behavior as four parts, 1) source, 2) target, 3) measuring
and 4) observation period. For example, an access behavior of an ad
request is the amount of appearance from an IP to a URL in the last one
hour, where the IP is source, the URL is target, the amount of
appearance is measuring and the last one hour is observation period. The
source and target information are in the incoming ad request and the
measuring is based on historical ad requests in the observation period.

To identify the source, we select three attributes as follows:

-   `audience_id`: In display advertising, the audience ID actually
    identifies the browser via third-party cookies. For a browser, the
    ID is renewed if the cookie is wiped.

-   `ip32`/`ip24`/`ip16`: We care not only the original IP, `ip32`, but
    also the associated subnet network, `ip24` and `ip16`. `ip32`,
    `ip24` and `ip16` represent the masked IP with 255.255.255.255,
    255.255.255.0 and 255.255.0.0, respectively.

-   `ua`: The user agent string identifies the kind of browser (such as,
    Chrome and Edge), browser version, OS (such as Windows and macOS)
    and OS version.

Moreover, we identify the target with different granularities. We select
five attributes as follows:

-   `url`: The URL in ad request.

-   `host`: The host name of the URL. One host name may contain many
    URLs.

-   `root_domain`: The root domain of the URL. One root domain may
    contain many hosts.

-   `space_id`: The space ID is to identify the ad placement or ad
    space, which is an area in website or in app for ad displaying.

-   `publisher_id`: The publisher ID is to identify the publisher. One
    publisher may contain many websites or apps.

For the measurement about access behavior, we adopt four directions
about non-intention characteristics according to the prior works. The
details are as follows:

-   The access amount: xxx [@11_IMC_Stone-Gross][@13_USS_Springborn].

-   The access diversity: xxx
    [@11_IMC_Stone-Gross][@14_JMLR_Oentaryo][@13_USS_Springborn].

-   The access frequency: xxx [@15_WWW_Tian][@14_JMLR_Oentaryo].

-   The click behavior: Due to high profit of click, some fraud
    activities aim at generating more clicks to make more profit for
    them
    [@11_IMC_Stone-Gross][@15_WWW_Tian][@14_JMLR_Oentaryo][@13_USS_Springborn].

The observation period xxx. xxx. hour and day.

  Measuring                                                                    Identifier                                       Period
  ---------------------------------------------------------------------------- ------------------------------------------------ ----------
  The amount of appearance                                                     `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Hour/Day
  The average/median/variance of inter-arrival time                            `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Hour/Day
  The amount/entropy of appeared hours                                         `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Day
  The amount/entropy of `ip32`/`ip24`/`ip16`                                   `audience_id`                                    Hour/Day
  The amount/entropy of `audience_id`                                          `ip32_ua`, `ip24_ua`, `ip16_ua`                  Hour/Day
  The amount/entropy of `url`/`host`/`root_domain`/`space_id`/`publisher_id`   `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Hour/Day
  The amount of distinct `url` divided by the amount of appearance             `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Hour/Day
  CTR                                                                          `ip32_ua`, `ip24_ua`, `ip16_ua`, `audience_id`   Hour/Day

  Measuring                                                                                Identifier                                                 Period
  ---------------------------------------------------------------------------------------- ---------------------------------------------------------- ----------
  The amount of appearance                                                                 `url`, `host`, `root_domain`, `space_id`, `publisher_id`   Hour/Day
  The amount/entropy of `ip32_ua`/`ip24_ua`/`ip16_ua`/`ip32`/`ip24`/`ip16`/`audience_id`   `url`, `host`, `root_domain`, `space_id`, `publisher_id`   Hour/Day
  CTR                                                                                      `url`, `host`, `root_domain`, `space_id`, `publisher_id`   Hour/Day

From tables [\[table:source\]](#table:source){reference-type="ref"
reference="table:source"} and
[\[table:target\]](#table:target){reference-type="ref"
reference="table:target"}, we had total 270 different features created
from audience and publisher, of which 180 from audience and 90 from
publisher.

Labeling
--------

It is challenge to label the non-intentional ad requests since there is
no ground truth. \[several method to detect bot, fraud \...\] \[no one
can exact detect non-intention\] \[in this paper, we adopt non-human
behavior which is human unreachable\] \[rule1 concept\] \[rule2
concept\]

Model Building
--------------

In model building stage, xxx

\[we adopt two algorithms from PU learning\] \[two-step concept\]
\[one-step concept\]

$D_P$, $D_U$ A classifier $f: \mathbb{R}^m \rightarrow \{0, 1\}$,
detected non-intentional ad requests $D'_P \subseteq D_U$

$count$ = 0 $sum_j^C$ = 0 $sum_j^U$ = 0

The $i$-th impression record $(\mathbf{x}_i$ =
$[x_{i1}, x_{i2}, \cdots, x_{im}], y_i)$ $sum_j^C$ += $x_{ij}$ $count$
+= 1 $sum_j^U$ += $x_{ij}$

$H$ = max_heap() $mean_j^C = sum_j^C / count$
$mean_j^U = sum_j^U / (m-count)$ $H$.push($j$, $|mean_j^C-mean_j^U|$)

$J$ = pop $m$ items from $H$

$J$

$D_P$, $D_U$ A classifier $f: \mathbb{R}^m \rightarrow \{0, 1\}$,
detected non-intentional ad requests $D'_P \subseteq D_U$

$count$ = 0 $sum_j^C$ = 0 $sum_j^U$ = 0

The $i$-th impression record $(\mathbf{x}_i$ =
$[x_{i1}, x_{i2}, \cdots, x_{im}], y_i)$ $sum_j^C$ += $x_{ij}$ $count$
+= 1 $sum_j^U$ += $x_{ij}$

$H$ = max_heap() $mean_j^C = sum_j^C / count$
$mean_j^U = sum_j^U / (m-count)$ $H$.push($j$, $|mean_j^C-mean_j^U|$)

$J$ = pop $m$ items from $H$

$J$

Performance Evaluation {#sec:evaluation}
======================

Dataset Description
-------------------

We use the real-world bid-request dataset to evaluate and test the
performance of our framework in inventory quality on demand-side in
display advertising. The full dataset is provided by TenMax AD Tech Lab
Co., LTD. without labeling in 3 days (07/01/2018 - 07/03/2018). Each
bid-request contains information such as time, audienceId, (IP, UA),
location, url, host, domain, etc. In that case, it is a challenge to
detect audience behaviors based on such information. Table
[1](#table:stat){reference-type="ref" reference="table:stat"} summarized
the basic statistics of our bid-request dataset

::: {#table:stat}
  Bid requests               265,994,333
  -------------------------- -------------
  Impressions                62,606,867
  Clicks                     82,379
  audienceIds                15,705,634
  (IP address, User Agent)   29,459,189
  CTR (Click through rate)   0.13 %

  : The basic statistics of dataset
:::

In order to adopt to machine learning algorithm, we spitted our data set
into two subsets: training set - the first 2 days and test set -- the
last day

Conclusion {#sec:conclusion}
==========

The purpose of the current study was to introduce the novel of detecting
non-intentional inventory on Demand-Side in display advertising. This
study has gone some way towards enhancing our understanding of the
practical challenge of ad fraud detection in display advertising,
especially in DSP side. The result of this work makes several
contributions to the current literature in detecting non-intentional
inventory quality on DSP. First, how to formulate the non-intentional
inventory identification as PU learning problem. Second, how to extract
the inventory access behavior and detect some non-intentional
inventories. Third, how to conduct the evaluation on real datasets. It
is unfortunate that the study did not show good results in classifying
intentional and non-intentional traffics since some main reasons were
mentioned in the discussion section. However, the finding of this study
has a number of important implications for the future practice of ad
fraud detection for DSP side in display advertising.

[^1]: The cost of a campaign divided by the number of clicks derived.

[^2]: For example, a URL is massively accessed by a browser from a IP in
    a short period.

[^3]: The typical ad types in display advertising are banner, native,
    video and so on.

[^4]: <https://www.iab.com/guidelines/real-time-bidding-rtb-project/>
