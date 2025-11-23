import { cityRecommendationApi, unifiedApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { months } from "@/app/utils/time";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { ResponsiveBar } from "@nivo/bar";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";
import customTheme from "@/app/data/theme.json";

const MAX_TOP_CITY_COUNT = 5;
const CityChart = ({ data }: { data: unifiedApi }) => {
  const { city_global_probabilities, market_matrix } = data.market_matrix;

  let chartData: { [k: string]: string | number }[] = [];
  for (const city in city_global_probabilities) {
    const prob = city_global_probabilities[city];
    if (prob > 0) {
      chartData.push({ City: city, Probability: prob });
    }
  }

  chartData = chartData
    .sort((a, b) => (b.Probability as number) - (a.Probability as number))
    .filter((_, i) => i < MAX_TOP_CITY_COUNT)
    .map((e) => ({ ...e, Probability: (e.Probability as number) / 100 }));

  for (const city in market_matrix) {
    const i = chartData.findIndex((e) => e.City === city);
    if (i !== -1) {
      const labelPercentage = chartData[i].Probability as number;
      const cityData = market_matrix[city];

      let sum = 0;
      for (const monthCount in cityData) {
        sum += cityData[monthCount];
      }

      let newChartElement: { [k: string]: string | number } = {
        City: city,
      };
      for (let i = 0; i < 12; i++) {
        const innerPercentage =
          (cityData[`Month_${i + 1}`] * labelPercentage) / sum;

        newChartElement[months[i]] = innerPercentage;
      }
      chartData[i] = newChartElement;
    }
  }

  return (
    <Card className="col-span-3">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start mb-5">City Recommendations</h3>

        <div className="w-full h-96">
          <ResponsiveBar /* or Bar for fixed dimensions */
            data={chartData}
            keys={[...months]}
            indexBy="City"
            valueFormat={">-.1%"}
            labelSkipWidth={12}
            enableTotals={true}
            enableLabel={false}
            legends={[
              {
                dataFrom: "keys",
                anchor: "bottom-right",
                direction: "column",
                translateX: 120,
                itemsSpacing: 3,
                itemWidth: 100,
                itemHeight: 16,
              },
            ]}
            colors={{ scheme: "spectral" }}
            labelSkipHeight={12}
            theme={customTheme}
            axisBottom={{ legend: "City", legendOffset: 32 }}
            axisLeft={{ legend: "Probability", legendOffset: -40 }}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
          />
        </div>
      </CardBody>
    </Card>
  );
};

export default CityChart;
