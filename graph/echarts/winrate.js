option = {
  color: ["#d8ead2", "#cfe2f5", "#F5F5F5"],
  tooltip: {
    trigger: "axis",
  },
  legend: {},
  xAxis: {
    type: "value",
  },
  yAxis: {
    type: "category",
    data: ["胜率"],
  },
  series: [
    {
      name: "胜",
      type: "bar",
      stack: "total",
      label: {
        show: true,
      },
      data: [42],
    },
    {
      name: "败",
      type: "bar",
      stack: "total",
      label: {
        show: true,
      },
      data: [20],
    },
    {
      name: "平",
      type: "bar",
      stack: "total",
      label: {
        show: true,
      },
      data: [38],
    },
  ],
};
