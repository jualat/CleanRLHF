import React from "react";
import {VideoGrid} from "~/components/video/video-grid";

export default function Page() {
  const videoData = [
    {
      title: "Video Title",
      description: "Description or additional information about the video goes here.",
      videoSrc: "/vids/InvertedDoublePendulum-v4__sac_rlhf__1__1734445231/rl-video-episode-0.mp4",
      poster: "https://via.placeholder.com/480",
    },
    {
      title: "Another Video",
      description: "Additional details or descriptions.",
      videoSrc: "/vids/InvertedDoublePendulum-v4__sac_rlhf__1__1734445231/rl-video-episode-0.mp4",
      poster: "https://via.placeholder.com/480",
    },
  ];

  return (
    <div className="container mx-auto py-8">
      <VideoGrid videos={videoData}/>
    </div>
  );
}