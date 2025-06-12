option = {
  xAxis: {
    type: "category",
    name: "文本长度",
    data: [4096, 8192, 16384, 32768, 65536, 131072],
  },
  yAxis: {
    name: "运行速度 token/s",
    type: "value",
  },
  legend: {},
  series: [
    {
      name: "Llama 3.1 8B",
      data: [26.62, 26.68, 25.06, 19.48, 12.31, 6.84],
      type: "bar",
      label: { show: true, position: "top" },
    },
    {
      name: "FalconMamba 7B",
      data: [13.61, 13.52, 13.45, 13.62, 13.54, 13.53],
      type: "bar",
      label: { show: true, position: "top" },
    },
  ],
};
