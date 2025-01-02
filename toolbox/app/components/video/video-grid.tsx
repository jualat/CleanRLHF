import React from "react";
import {VideoCard} from "~/components/video/video-card";

export const VideoGrid = ({ videos }: {
  videos: {
    title: string;
    description: string;
    videoSrc: string;
    poster: string;
  }[];
}) => {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-2xl font-bold text-center mb-6">Video Gallery</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {videos.map((video, index) => (
          <VideoCard
            key={index}
            title={video.title}
            description={video.description}
            videoSrc={video.videoSrc}
            poster={video.poster}
          />
        ))}
      </div>
    </div>
  );
};