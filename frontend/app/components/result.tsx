"use client";

import CityChart from "./charts/city-chart";
import FeedbackChart from "./charts/feedback-chart";
import MonthChart from "./charts/month-chart";
import RatingChart from "./charts/rating-chart";
import SuccessChart from "./charts/success-chart";
import UserDetails from "./charts/user-details";

const Result = () => {
  return (
    <div className="grid grid-cols-6 gap-2 grid-flow-dense">
      <UserDetails />
      <FeedbackChart />
      <SuccessChart />
      <RatingChart />
      <CityChart />
      <MonthChart />
    </div>
  );
};

export default Result;
