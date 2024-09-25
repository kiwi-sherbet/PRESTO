---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>PRESTO: Fast motion planning using diffusion models based on key-configuration environment representation</title>

<!-- <meta property="og:image" content="src/figure/approach.png"> -->
<meta property="og:title" content="PRESTO">

<script src="./src/popup.js" type="text/javascript"></script>
<script src="/src/js/viewstl/stl_viewer.min.js"></script>
<script src="/src/js/viewstl/init.js"></script>

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-TDC3EZ02LM"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-TDC3EZ02LM');
</script>

<script>
  document.addEventListener('DOMContentLoaded', function() {
    const videos = document.querySelectorAll('video.lazy-video');
    
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.play();
        } else {
          entry.target.pause();
        }
      });
    }, {
      threshold: 0.5 // Adjust this as needed (0.5 means 50% of the video must be visible)
    });
    
    videos.forEach(video => {
      observer.observe(video);
    });
  });
</script>

<!-- STLviewer tag -->
<!-- <script>
function initStlViewer() {
    var $modelElements = $("div.3d-model");
    $modelElements.each(function (i, elem) {
        var filePath = $(elem).data('src');
        console.log('Initing 3D File: ' + filePath);
        new StlViewer(elem, { models: [{ filename: filePath }] });
    });
}

$(document).ready(initStlViewer);
</script> -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
h1 { 
    font-weight:300; 
  }
h2 {
    font-weight:300;
    font-size:24px;
  }
h3 {
    font-weight:300;
}
IMG {
    PADDING-RIGHT: 0px;
    PADDING-LEFT: 0px;
    <!-- FLOAT: justify; -->
    PADDING-BOTTOM: 0px;
    PADDING-TOP: 0px;
    display:block;
    margin:auto;  
  }
#primarycontent {
    MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
    1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
    1000px 
  }
BODY {
    TEXT-ALIGN: center
}
hr{
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }
pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
  }
table {
  	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">


<style>
a {
  color: #800080;
  text-decoration: none;
  font-weight: 500;
}
</style>


<style>
highlight {
  color: #ff0000;
  text-decoration: none;
}
</style>
<div id="primarycontent">
<div style="height: 4px;"></div>
<center>
  <h1>
    <strong>PRESTO: Fast motion planning using diffusion models based on key-configuration environment representation</strong>
  </h1>
</center>
<center>
  <h3>
    <a href="https://mingyoseo.com/">Mingyo Seo<sup>1*</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://yycho0108.github.io/research">Yoonyoung Cho<sup>2*</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://yoonchangsung.com/">Yoonchang Sung<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.cs.utexas.edu/~pstone/">Peter Stone<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://yukezhu.me/">Yuke Zhu<sup>1&dagger;</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://beomjoonkim.github.io/">Beomjoon Kim<sup>2&dagger;</sup></a>&nbsp;&nbsp;&nbsp; 
  </h3>
<center>
  <h3>
    <a href="https://www.utexas.edu/"><sup>1</sup>The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.kaist.ac.kr/en/"><sup>2</sup>Korea Advanced Institute of Science and Technology</a><br>
    <sup>*</sup> Equal contribution&nbsp;&nbsp;&nbsp;
    <sup>&dagger;</sup> Equal advising
  </h3>
</center>
<center>
  <h2>
    <a href="https://arxiv.org/abs/2409.16012">Paper</a> | <a href="./src/file/appendix.pdf" download>Appendix</a>
  </h2>
</center>

<center>
  <p>
    <span style="font-size:20px;"></span>
  </p>
</center>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
    <tr>
      <td align="center" valign="middle">
        <video muted autoplay loop width="798">
          <source src="./src/video/header.mp4"  type="video/mp4">
        </video>
      </td>
    </tr> 
  </tbody> 
</table>

<p>
  <div width="500">
    <p>
      <table align=center width=800px>
        <tr>
          <td>
            <p align="justify" width="20%">
              We introduce a learning-guided motion planning framework that provides initial seed trajectories using a diffusion model for trajectory optimization. Given a workspace, our method approximates the configuration space (C-space) obstacles through a key-configuration representation that consists of a sparse set of task-related key configurations, and uses this as an input to the diffusion model. The diffusion model integrates regularization terms that encourage collision avoidance and smooth trajectories during training, and trajectory optimization refines the generated seed trajectories to further correct any colliding segments. Our experimental results demonstrate that using high-quality trajectory priors, learned through our C-space-grounded diffusion model, enables efficient generation of collision-free trajectories in narrow-passage environments, outperforming prior learning- and planning-based baselines.
      	    </p>
          </td>
        </tr>
      </table>
    </p>
  </div>
</p>

<hr>
<h1 align="center">Main Results</h1>

  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="190">
            <source src="./src/video/level1.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="190">
            <source src="./src/video/level2.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="190">
            <source src="./src/video/level3.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="190">
            <source src="./src/video/level4.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
    </tbody>
  </table>
  <table align=center width=800px><tr><td><p align="justify" width="20%">
  We evaluate our method on a motion planning task using the Franka Emika Panda robot arm to traverse a 3-tier shelf with various objects in simulation. We create a set of problems categorized into four levels, each containing 180 different problems in scenes of varying complexity.
  </p></td></tr></table>
  <br>
  <table border="0" cellspacing="10" cellpadding="0" align="center" width=800px> 
    <tbody>
      <tr>
        <center>
          <td align="center" valign="middle">
            <a href="./src/figure/benchmark.svg"><img src="./src/figure/benchmark.svg" style="width:130%; margin-left:-15%;"> </a>
          </td>
        </center>
      </tr>
    </tbody>
  </table>
  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="258">
            <source src="./src/video/birrt.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="258">
            <source src="./src/video/trajopt.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="258">
            <source src="./src/video/presto.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
      <tr><td></td><td></td><td><p align="right">
	  <font color="blue">Collision-free</font>/<font color="red">Colliding</font>
      </p></td></tr>
    </tbody>
  </table>
  <table align=center width=800px><tr><td><p align="justify" width="20%">
  Across all levels, PRESTO consistently outperforms the pure learning algorithms <a href="https://scenediffuser.github.io/">SceneDiffuser</a> and <a href="https://arxiv.org/abs/2308.01557">Motion Planning Diffuser (MPD)</a>, which lack a key-configuration-based environment representation and a motion-planning-based objective, respectively. Compared to Bi-RRT, PRESTO uses diffusion-learned trajectory priors to generate collision-free trajectories more efficiently, especially in narrow passages. Additionally, compared to <a href="https://rll.berkeley.edu/trajopt/doc/sphinx_build/html/">TrajOpt</a>, an optimization-based method, PRESTO's high-quality initial trajectories lead to faster convergence in complex domains, despite the computational overhead of running the diffusion model.
  </p></td></tr></table>
<hr>

<h1 align="center">Ablation Studies</h1>

  <table border="0" cellspacing="10" cellpadding="0" align="center" width=800px> 
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <a href="./src/figure/ablation.svg"><img src="./src/figure/ablation.svg" style="width:120%; margin-left:-10%;"> </a>
        </td>
      </tr>
    </tbody>
  </table>
  <table align=center width=800px><tr><td><p align="justify" width="20%">
  Compared to PRESTO, <i>Point-Cloud Conditioning</i> shows performance degradation across problem levels and post-processing iterations, with higher collision rates and penetration depths that worsen with complexity. Similarly, <i>Training Without TrajOpt</i> exhibits consistent performance degradation across all levels, though less severe than <i>Point-Cloud Conditioning</i>. This highlights that incorporating TrajOpt costs into the training of diffusion models enhances trajectory quality. Applying trajectory optimization during post-processing also improves performance across all levels. Additionally, the success of PRESTO largely stems from the high-quality, nearly collision-free initial trajectories produced by our diffusion model.
  </p></td></tr></table>
  <br>
  <table border="0" cellspacing="10" cellpadding="0" align="center" width=800px> 
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <a href="./src/figure/guidance.svg"><img src="./src/figure/guidance.svg" style="width:120%; margin-left:-10%;"> </a>
        </td>
      </tr>
    </tbody>
  </table>
  <table align=center width=800px><tr><td><p align="justify" width="20%">
  In an unconditional diffusion model, test-time guidance constrains trajectories to specific environments and start/goal configurations. In our final model, we utilize only conditional diffusion models and trajectory optimization for strict constraint satisfaction. Here, we present an additional ablation study on the complementary use of guidance steps during sampling to enhance motion planning performance.
  Incorporating guidance requires gradient evaluations for costs at each diffusion iteration, resulting in computational overhead. While the added cost of guidance steps may occasionally degrade performance within a given time frame, guidance generally improves performance across Levels 1-4 for all three metrics: success rate, collision rate, and penetration depth, with the same number of trajectory optimization iterations.
  </p></td></tr></table>

<hr>
<center><h1>Citation</h1></center>

<table align=center width=800px>
  <tr>
    <td>
    <!-- <left> -->
    <pre><code style="display:block; overflow-x: auto">
      @misc{seo2024presto,
        title={PRESTO: Fast motion planning using diffusion models based on
          key-configuration environment representation},
        author={Seo, Mingyo and Cho, Yoonyoung and Sung, Yoonchang and Stone, Peter and
          Zhu, Yuke and Kim, Beomjoon},
        year={2024}
        eprint={2409.16012},
        archivePrefix={arXiv},
        primaryClass={cs.RO}
      }
    </code></pre>
    <!-- </left> -->
    </td>
  </tr>
</table>
