import { monthRecommendationApi, unifiedApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";
import { ResponsiveLine } from "@nivo/line";
import { months } from "@/app/utils/time";
import customTheme from "@/app/data/theme.json";

const MonthChart = ({ data }: { data: unifiedApi }) => {
  const { market_matrix, city_global_probabilities } = data.market_matrix;

  let chartData = [
    { id: "Month", data: months.map((month) => ({ x: month, y: 0 })) },
  ];

  for (const city in market_matrix) {
    const cityData = market_matrix[city];
    let sum = 0;
    for (const month in cityData) {
      sum += cityData[month];
    }
    if (sum === 0) continue;

    console.log(city, sum);
    for (let i = 0; i < 12; i++) {
      console.log(cityData[`Month_1`]);

      chartData[0].data[i].y += cityData[`Month_${i + 1}`] / sum;
    }
  }

  let sum = 0;
  for (const dataPoint of chartData[0].data) {
    sum += dataPoint.y;
  }
  for (let dataPoint of chartData[0].data) {
    dataPoint.y /= sum;
  }

  return (
    <Card className="col-span-3">
      <CardBody className="flex flex-col items-center">
        <h3 className="self-start mb-5">Month Recommendations</h3>

        <div className="w-full h-96">
          <ResponsiveLine /* or Line for fixed dimensions */
            data={chartData}
            margin={{ top: 50, right: 50, bottom: 50, left: 60 }}
            yScale={{
              type: "linear",
              min: "auto",
              max: "auto",
              stacked: true,
              reverse: false,
            }}
            curve="catmullRom"
            enableArea={true}
            axisBottom={{ legend: "Months", legendOffset: 36 }}
            axisLeft={{ legend: "Probability", legendOffset: -40 }}
            pointSize={10}
            yFormat={">-.1%"}
            colors={{ scheme: "category10" }}
            pointColor={{ theme: "background" }}
            pointBorderWidth={2}
            pointBorderColor={{ from: "seriesColor" }}
            pointLabelYOffset={-12}
            theme={customTheme}
            enableTouchCrosshair={true}
            useMesh={true}
          />
        </div>
      </CardBody>
    </Card>
  );
};

export default MonthChart;
