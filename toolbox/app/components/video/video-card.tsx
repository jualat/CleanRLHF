import React from "react";

export const VideoCard = ({ title, description, videoSrc, poster }: {
  title: string;
  description: string;
  videoSrc: string;
  poster: string;
}) => {
  return (
    <div className="group bg-white shadow-md rounded-lg overflow-hidden hover:shadow-lg">
      <div className="relative aspect-square">
        <video
          className="w-full h-full object-cover"
          controls
          autoPlay={true}
          poster={poster}
        >
          <source src={videoSrc} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition">
          <button className="text-white bg-gray-800 bg-opacity-75 px-4 py-2 rounded-lg">Play Video</button>
        </div>
      </div>
      <div className="p-4">
        <h2 className="text-lg font-medium text-gray-800">{title}</h2>
        <p className="text-sm text-gray-600 mt-2">{description}</p>
      </div>
    </div>
  );
};
